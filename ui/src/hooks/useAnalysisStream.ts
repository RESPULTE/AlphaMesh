import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  AgentAnalysis,
  AnalysisResponse,
  DataFramePayload,
  FinalResult,
  StreamEvent
} from '../types/api';

const DEFAULT_USER_EMAIL = 'demo@alphamesh.local';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';
const STORAGE_SESSION_ID = 'alphamesh.active_session_id';

type PartialAnalysis = Partial<AnalysisResponse> | null;

function emptySummary() {
  return {
    coreNarrative: '',
    agentConsensus: [],
    verdict: { label: '', description: '' }
  };
}

function placeholderAgents(): AgentAnalysis[] {
  return [
    {
      id: 'news',
      name: 'News Analysis Agent',
      icon: 'news',
      category: 'Intelligence Unit',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      fullReport: '',
      references: []
    },
    {
      id: 'fundamental',
      name: 'Fundamental Agent',
      icon: 'analytics',
      category: 'Financial Lab',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      metrics: [],
      quote: '',
      fullReport: ''
    }
  ];
}

function baseResponse(): AnalysisResponse {
  return {
    ticker: '',
    companyName: '',
    currentPrice: null,
    priceChange: null,
    priceChangePercent: null,
    marketStatus: 'MARKET DATA UNAVAILABLE',
    chartData: [],
    agents: placeholderAgents(),
    summary: emptySummary()
  };
}

function extractDomain(url: string): string {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, '');
  } catch {
    return 'Source';
  }
}

function firstSentence(text: string): string {
  const match = text.match(/^(.*?[.!?])\s/);
  return match ? match[1] : text.slice(0, 160);
}

function deriveConsensus(agentAnalyses: Record<string, string>): AnalysisResponse['summary']['agentConsensus'] {
  const consensus: AnalysisResponse['summary']['agentConsensus'] = [];
  if (agentAnalyses.news_agent) {
    consensus.push({
      title: 'News Sentiment',
      description: 'Recent media coverage and event catalysts assessed.',
      icon: 'verified'
    });
  }
  if (agentAnalyses.fundamentals_agent) {
    consensus.push({
      title: 'Fundamental Strength',
      description: 'Financial statements and quantitative metrics evaluated.',
      icon: 'account_balance'
    });
  }
  return consensus;
}

function formatValue(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toFixed(2);
}

function buildTable(payload?: DataFramePayload | null): AgentAnalysis['tableData'] | undefined {
  if (!payload || payload.data.length === 0) return undefined;
  const maxRows = Math.min(8, payload.index.length);
  const maxCols = Math.min(5, payload.columns.length);
  const columnStart = payload.columns.length - maxCols;
  const headers = ['Metric', ...payload.columns.slice(columnStart)];
  const rows: string[][] = [];
  for (let r = 0; r < maxRows; r++) {
    const rowLabel = payload.index[r];
    const rowValues = payload.data[r]
      .slice(columnStart)
      .map((val) => (val == null ? '�' : formatValue(val)));
    rows.push([rowLabel, ...rowValues]);
  }
  return { title: 'Financial Data', headers, rows };
}

function buildMetrics(payload?: DataFramePayload | null): AgentAnalysis['metrics'] | undefined {
  if (!payload || payload.data.length === 0) return undefined;
  const lastColIndex = payload.columns.length - 1;
  const metrics: AgentAnalysis['metrics'] = [];
  for (let i = 0; i < payload.index.length && metrics.length < 4; i++) {
    const val = payload.data[i]?.[lastColIndex];
    if (val == null) continue;
    metrics.push({ label: payload.index[i].toUpperCase().slice(0, 12), value: formatValue(val) });
  }
  return metrics.length ? metrics : undefined;
}

function updateFundamentalAgent(prev: AnalysisResponse, payload: DataFramePayload): AnalysisResponse {
  const metrics = buildMetrics(payload);
  const tableData = buildTable(payload);
  if (!metrics && !tableData) return prev;

  const agents = prev.agents.length ? [...prev.agents] : placeholderAgents();
  const idx = agents.findIndex((agent) => agent.id === 'fundamental');
  if (idx === -1) {
    agents.push({
      id: 'fundamental',
      name: 'Fundamental Agent',
      icon: 'analytics',
      category: 'Financial Lab',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      metrics,
      tableData,
      quote: ''
    });
  } else {
    agents[idx] = {
      ...agents[idx],
      metrics: metrics ?? agents[idx].metrics,
      tableData: tableData ?? agents[idx].tableData
    };
  }

  return { ...prev, agents };
}

function mapFinalResult(result: FinalResult): AnalysisResponse {
  const response = baseResponse();
  const tickerResult = result.ticker_results[0];
  if (tickerResult) {
    response.ticker = tickerResult.ticker || '';
    response.companyName = tickerResult.ticker || '';
  }

  const agentAnalyses = result.agent_analyses || {};
  const agents: AgentAnalysis[] = [];

  if (agentAnalyses.news_agent || tickerResult?.sources?.length) {
    const text = agentAnalyses.news_agent || tickerResult?.analysis_text || result.synthesis || '';
    const catalystTitle = firstSentence(text);
    agents.push({
      id: 'news',
      name: 'News Analysis Agent',
      icon: 'news',
      category: 'Intelligence Unit',
      recentCatalyst: {
        title: catalystTitle,
        description: text.slice(0, 140),
        timeAgo: 'RECENT'
      },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      fullReport: text,
      references: (tickerResult?.sources || []).map((s) => ({
        id: s.source_id,
        title: s.title,
        url: s.url,
        source: extractDomain(s.url)
      }))
    });
  }

  if (agentAnalyses.fundamentals_agent || tickerResult?.financial_data) {
    const text = agentAnalyses.fundamentals_agent || tickerResult?.analysis_text || '';
    const tableData = buildTable(tickerResult?.financial_data);
    agents.push({
      id: 'fundamental',
      name: 'Fundamental Agent',
      icon: 'analytics',
      category: 'Financial Lab',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      metrics: buildMetrics(tickerResult?.financial_data),
      quote: firstSentence(text),
      fullReport: text,
      tableData
    });
  }

  response.agents = agents.length ? agents : placeholderAgents();
  response.summary = {
    coreNarrative: result.synthesis || tickerResult?.analysis_text || '',
    agentConsensus: deriveConsensus(agentAnalyses),
    verdict: {
      label: 'NEUTRAL',
      description: firstSentence(result.synthesis || tickerResult?.analysis_text || '')
    }
  };

  return response;
}

function mergeFinalWithLive(next: AnalysisResponse, prev?: AnalysisResponse | null): AnalysisResponse {
  if (!prev) return next;
  return {
    ...next,
    ticker: next.ticker || prev.ticker,
    companyName: next.companyName || prev.companyName,
    currentPrice: next.currentPrice ?? prev.currentPrice,
    priceChange: next.priceChange ?? prev.priceChange,
    priceChangePercent: next.priceChangePercent ?? prev.priceChangePercent,
    marketStatus: next.marketStatus || prev.marketStatus,
    chartData: next.chartData.length ? next.chartData : prev.chartData,
    agents: next.agents.length ? next.agents : prev.agents
  };
}

export function useAnalysisStream(query: string | null) {
  const [data, setData] = useState<PartialAnalysis>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const conversationIdRef = useRef<string | null>(
    typeof window !== 'undefined'
      ? window.localStorage.getItem(STORAGE_CONVERSATION_ID)
      : null
  );
  const sessionIdRef = useRef<string | null>(
    typeof window !== 'undefined'
      ? window.localStorage.getItem(STORAGE_SESSION_ID)
      : null
  );
  const resolvedTickerRef = useRef<string>('');
  const pendingQuoteRef = useRef<AnalysisResponse | null>(null);
  const pendingChartRef = useRef<AnalysisResponse['chartData'] | null>(null);

  const stableEmail = useMemo(() => DEFAULT_USER_EMAIL, []);

  useEffect(() => {
    if (!query) return;

    let isMounted = true;
    setIsStreaming(true);
    setData(baseResponse());
    resolvedTickerRef.current = '';
    pendingQuoteRef.current = null;
    pendingChartRef.current = null;

    const closeStream = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };

    const start = async () => {
      try {
        const res = await fetch('/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: query,
            user_email: stableEmail,
            conversation_id: conversationIdRef.current,
            session_id: sessionIdRef.current
          })
        });

        if (!res.ok) {
          throw new Error(`Chat request failed: ${res.status}`);
        }

        const ack = await res.json();
        if (!isMounted) return;
        if (ack.conversation_id) {
          conversationIdRef.current = ack.conversation_id;
          if (typeof window !== 'undefined') {
            window.localStorage.setItem(STORAGE_CONVERSATION_ID, ack.conversation_id);
          }
        }
        if (ack.session_id) {
          sessionIdRef.current = ack.session_id;
          if (typeof window !== 'undefined') {
            window.localStorage.setItem(STORAGE_SESSION_ID, ack.session_id);
          }
        }

        const es = new EventSource(`/api/v1/stream/${ack.request_id}`);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
          if (!isMounted) return;
          let payload: StreamEvent;
          try {
            payload = JSON.parse(event.data) as StreamEvent;
          } catch {
            return;
          }

          if (payload.event_type === 'progress') {
            setIsStreaming(true);
            return;
          }

          if (payload.event_type === 'init') {
            const resolvedTicker = resolvedTickerRef.current;
            if (!resolvedTicker) {
              pendingQuoteRef.current = {
                ...(pendingQuoteRef.current ?? baseResponse()),
                ticker: payload.quote.ticker ?? '',
                companyName: payload.quote.companyName ?? '',
                currentPrice: payload.quote.currentPrice ?? null,
                priceChange: payload.quote.priceChange ?? null,
                priceChangePercent: payload.quote.priceChangePercent ?? null,
                marketStatus: payload.quote.marketStatus ?? 'MARKET DATA UNAVAILABLE'
              };
              return;
            }

            if ((payload.quote.ticker || '').toUpperCase() !== resolvedTicker) {
              return;
            }

            setData((prev) => {
              const current = prev ?? baseResponse();
              return {
                ...current,
                ticker: payload.quote.ticker ?? current.ticker,
                companyName: payload.quote.companyName ?? current.companyName,
                currentPrice: payload.quote.currentPrice ?? current.currentPrice,
                priceChange: payload.quote.priceChange ?? current.priceChange,
                priceChangePercent: payload.quote.priceChangePercent ?? current.priceChangePercent,
                marketStatus: payload.quote.marketStatus ?? current.marketStatus
              };
            });
            return;
          }

          if (payload.event_type === 'chart') {
            const resolvedTicker = resolvedTickerRef.current;
            if (!resolvedTicker) {
              pendingChartRef.current = payload.chart ?? [];
              return;
            }
            setData((prev) => (prev ? { ...prev, chartData: payload.chart ?? [] } : prev));
            return;
          }
          if (payload.event_type === 'ticker_resolved') {
            const resolved = (payload.ticker || payload.tickers?.[0] || '').toUpperCase();
            if (resolved) {
              resolvedTickerRef.current = resolved;
              const pendingQuote = pendingQuoteRef.current;
              if (
                pendingQuote &&
                (pendingQuote.ticker || '').toUpperCase() === resolved
              ) {
                setData((prev) => {
                  const current = prev ?? baseResponse();
                  return {
                    ...current,
                    ticker: pendingQuote.ticker || current.ticker,
                    companyName: pendingQuote.companyName || current.companyName,
                    currentPrice: pendingQuote.currentPrice ?? current.currentPrice,
                    priceChange: pendingQuote.priceChange ?? current.priceChange,
                    priceChangePercent:
                      pendingQuote.priceChangePercent ?? current.priceChangePercent,
                    marketStatus: pendingQuote.marketStatus || current.marketStatus
                  };
                });
              }

              const pendingChart = pendingChartRef.current;
              if (pendingChart && pendingChart.length) {
                setData((prev) => (prev ? { ...prev, chartData: pendingChart } : prev));
              }
            }
            return;
          }
          if (payload.event_type === 'metrics') {
            setData((prev) => {
              if (!prev) return prev;
              return updateFundamentalAgent(prev as AnalysisResponse, payload.financial_data);
            });
            return;
          }

          if (payload.event_type === 'complete' && payload.result) {
            const mapped = mapFinalResult(payload.result);
            const finalTicker = (payload.result.ticker_results?.[0]?.ticker || '').toUpperCase();
            resolvedTickerRef.current = finalTicker;

            if (finalTicker) {
              const pendingQuote = pendingQuoteRef.current;
              if (
                pendingQuote &&
                (pendingQuote.ticker || '').toUpperCase() === finalTicker
              ) {
                setData((prev) => {
                  const current = prev ?? baseResponse();
                  return {
                    ...current,
                    ticker: pendingQuote.ticker || current.ticker,
                    companyName: pendingQuote.companyName || current.companyName,
                    currentPrice: pendingQuote.currentPrice ?? current.currentPrice,
                    priceChange: pendingQuote.priceChange ?? current.priceChange,
                    priceChangePercent:
                      pendingQuote.priceChangePercent ?? current.priceChangePercent,
                    marketStatus: pendingQuote.marketStatus || current.marketStatus
                  };
                });
              }

              const pendingChart = pendingChartRef.current;
              if (pendingChart && pendingChart.length) {
                setData((prev) => (prev ? { ...prev, chartData: pendingChart } : prev));
              }
            }
            setData((prev) => mergeFinalWithLive(mapped, prev as AnalysisResponse));
            setIsStreaming(false);
            closeStream();
          }

          if (payload.event_type === 'error') {
            setIsStreaming(false);
            closeStream();
          }
        };

        es.onerror = () => {
          if (!isMounted) return;
          setIsStreaming(false);
          closeStream();
        };
      } catch {
        if (!isMounted) return;
        setIsStreaming(false);
      }
    };

    start();

    return () => {
      isMounted = false;
      closeStream();
    };
  }, [query, stableEmail]);

  return { data, isStreaming };
}
