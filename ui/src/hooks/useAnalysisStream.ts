import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import {
  buildFundamentalsMetrics,
  buildFundamentalsTable,
} from '../lib/fundamentalsDisplay';
import type {
  AgentAnalysis,
  AnalysisResponse,
  ConversationSummary,
  DataFramePayload,
  FinalResult,
  FundamentalsVisualizationPayload,
  StreamEvent
} from '../types/api';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';
const STORAGE_SESSION_ID = 'alphamesh.active_session_id';

type PartialAnalysis = Partial<AnalysisResponse> | null;
export type StreamPhase =
  | 'idle'
  | 'starting'
  | 'awaiting_ticker'
  | 'streaming'
  | 'completed'
  | 'error';

function emptySummary() {
  return {
    coreNarrative: '',
    agentConsensus: [],
    verdict: { label: '', description: '' }
  };
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
    fundamentalData: null,
    fundamentalsVisualization: null,
    agents: [],
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

function updateFundamentalAgent(prev: AnalysisResponse, payload: DataFramePayload): AnalysisResponse {
  const metrics = buildFundamentalsMetrics(payload);
  const tableData = buildFundamentalsTable(payload);
  if (!metrics && !tableData) return prev;

  const agents = [...prev.agents];
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

  return { ...prev, agents, fundamentalData: payload };
}

function mergeVisualizationPayload(
  prev: FundamentalsVisualizationPayload | null | undefined,
  next: FundamentalsVisualizationPayload | null | undefined
): FundamentalsVisualizationPayload | null {
  if (!prev && !next) return null;
  if (!prev) return next ?? null;
  if (!next) return prev;
  return {
    charts: next.charts?.length ? next.charts : prev.charts,
    raw_row_labels: next.raw_row_labels?.length ? next.raw_row_labels : prev.raw_row_labels,
    raw_data: next.raw_data ?? prev.raw_data,
    reviewer_notes: next.reviewer_notes || prev.reviewer_notes,
    task_completed: next.task_completed ?? prev.task_completed,
    task_completion_reason: next.task_completion_reason || prev.task_completion_reason
  };
}

function updateFundamentalVisualization(
  prev: AnalysisResponse,
  payload: FundamentalsVisualizationPayload
): AnalysisResponse {
  const mergedVisualization = mergeVisualizationPayload(prev.fundamentalsVisualization, payload);
  const tablePayload = payload.raw_data ?? prev.fundamentalData;
  const metrics = buildFundamentalsMetrics(tablePayload);
  const tableData = buildFundamentalsTable(tablePayload);
  const agents = [...prev.agents];
  const idx = agents.findIndex((agent) => agent.id === 'fundamental');

  if (idx >= 0) {
    agents[idx] = {
      ...agents[idx],
      metrics: metrics ?? agents[idx].metrics,
      tableData: tableData ?? agents[idx].tableData
    };
  }

  return {
    ...prev,
    fundamentalsVisualization: mergedVisualization,
    fundamentalData: tablePayload ?? prev.fundamentalData,
    agents
  };
}

function mapFinalResult(result: FinalResult): AnalysisResponse {
  const response = baseResponse();
  const tickerResult = result.ticker_results[0];
  if (tickerResult) {
    response.ticker = tickerResult.ticker || '';
    response.companyName =
      tickerResult.market_quote?.companyName || tickerResult.ticker || '';
    response.currentPrice = tickerResult.market_quote?.currentPrice ?? null;
    response.priceChange = tickerResult.market_quote?.priceChange ?? null;
    response.priceChangePercent = tickerResult.market_quote?.priceChangePercent ?? null;
    response.marketStatus =
      tickerResult.market_quote?.marketStatus || response.marketStatus;
    response.chartData = tickerResult.market_chart ?? [];
    response.fundamentalData = tickerResult.financial_data ?? null;
    response.fundamentalsVisualization = tickerResult.fundamentals_visualization ?? null;
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
    const tableData = buildFundamentalsTable(tickerResult?.financial_data);
    agents.push({
      id: 'fundamental',
      name: 'Fundamental Agent',
      icon: 'analytics',
      category: 'Financial Lab',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      metrics: buildFundamentalsMetrics(tickerResult?.financial_data),
      quote: firstSentence(text),
      fullReport: text,
      tableData
    });
  }

  response.agents = agents;
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
    fundamentalData: next.fundamentalData ?? prev.fundamentalData ?? null,
    fundamentalsVisualization: mergeVisualizationPayload(
      prev.fundamentalsVisualization,
      next.fundamentalsVisualization
    ),
    agents: next.agents.length ? next.agents : prev.agents
  };
}

export function useAnalysisStream(query: string | null, requestVersion = 0) {
  const { authFetch, buildAuthUrl } = useAuth();
  const [data, setData] = useState<PartialAnalysis>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamPhase, setStreamPhase] = useState<StreamPhase>('idle');
  const [conversationId, setConversationId] = useState<string | null>(
    typeof window !== 'undefined'
      ? window.localStorage.getItem(STORAGE_CONVERSATION_ID)
      : null
  );
  const [requestId, setRequestId] = useState<string | null>(null);
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
  const pendingFundamentalsVisualizationRef = useRef<FundamentalsVisualizationPayload | null>(null);

  useEffect(() => {
    if (!query) return;

    let isMounted = true;
    setIsStreaming(true);
    setStreamPhase('starting');
    setRequestId(null);
    setData(baseResponse());
    resolvedTickerRef.current = '';
    pendingQuoteRef.current = null;
    pendingChartRef.current = null;
    pendingFundamentalsVisualizationRef.current = null;

    const closeStream = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };

    const start = async () => {
      try {
        if (!conversationIdRef.current) {
          try {
            const latestRes = await authFetch('/api/v1/conversations?limit=1');
            if (latestRes.ok) {
              const rows = (await latestRes.json()) as ConversationSummary[];
              const latestConversationId = rows?.[0]?.conversation_id;
              if (latestConversationId) {
                conversationIdRef.current = latestConversationId;
                setConversationId(latestConversationId);
                if (typeof window !== 'undefined') {
                  window.localStorage.setItem(STORAGE_CONVERSATION_ID, latestConversationId);
                }
              }
            }
          } catch {
            // best-effort only; fallback is creating a new conversation on POST /chat
          }
        }

        const res = await authFetch('/api/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: query,
            conversation_id: conversationIdRef.current,
            session_id: sessionIdRef.current
          })
        });

        if (!res.ok) {
          throw new Error(`Chat request failed: ${res.status}`);
        }

        const ack = await res.json();
        if (!isMounted) return;
        if (ack.request_id) {
          setRequestId(ack.request_id);
        }
        if (ack.conversation_id) {
          conversationIdRef.current = ack.conversation_id;
          setConversationId(ack.conversation_id);
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

        const es = new EventSource(buildAuthUrl(`/api/v1/stream/${ack.request_id}`));
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
            setStreamPhase((prev) => (prev === 'awaiting_ticker' ? prev : 'streaming'));
            return;
          }

          if (payload.event_type === 'init') {
            const resolvedTicker = resolvedTickerRef.current;
            if (!resolvedTicker) {
              setStreamPhase('awaiting_ticker');
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
              setStreamPhase('awaiting_ticker');
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
              setStreamPhase('streaming');
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

          if (payload.event_type === 'fundamentals_visualization') {
            pendingFundamentalsVisualizationRef.current = payload.fundamentals_visualization;
            setData((prev) => {
              if (!prev) return prev;
              return updateFundamentalVisualization(
                prev as AnalysisResponse,
                payload.fundamentals_visualization
              );
            });
            return;
          }

          if (payload.event_type === 'complete' && payload.result) {
            const mapped = mapFinalResult(payload.result);
            const finalTicker = (payload.result.ticker_results?.[0]?.ticker || '').toUpperCase();
            resolvedTickerRef.current = finalTicker;
            const incrementalVisualization = pendingFundamentalsVisualizationRef.current;
            const finalVisualization =
              payload.result.ticker_results?.[0]?.fundamentals_visualization ??
              incrementalVisualization;
            mapped.fundamentalsVisualization = mergeVisualizationPayload(
              mapped.fundamentalsVisualization,
              finalVisualization
            );
            if (!mapped.fundamentalData && finalVisualization?.raw_data) {
              mapped.fundamentalData = finalVisualization.raw_data;
            }

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
            setData((prev) => {
              const merged = mergeFinalWithLive(mapped, prev as AnalysisResponse);
              if (finalVisualization) {
                return updateFundamentalVisualization(merged, finalVisualization);
              }
              return merged;
            });
            setIsStreaming(false);
            setStreamPhase('completed');
            closeStream();
          }

          if (payload.event_type === 'error') {
            setIsStreaming(false);
            setStreamPhase('error');
            closeStream();
          }
        };

        es.onerror = () => {
          if (!isMounted) return;
          setIsStreaming(false);
          setStreamPhase('error');
          closeStream();
        };
      } catch {
        if (!isMounted) return;
        setIsStreaming(false);
        setStreamPhase('error');
      }
    };

    start();

    return () => {
      isMounted = false;
      closeStream();
    };
  }, [authFetch, buildAuthUrl, query, requestVersion]);

  return { data, isStreaming, streamPhase, conversationId, requestId };
}
