import type {
  AgentAnalysis,
  AnalysisResponse,
  ConversationTurn,
  DataFramePayload,
} from '../types/api';

function firstSentence(text: string): string {
  const match = text.match(/^(.*?[.!?])\s/);
  return match ? match[1] : text.slice(0, 160);
}

function extractDomain(url: string): string {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, '');
  } catch {
    return 'Source';
  }
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
      .map((val) => (val == null ? '-' : formatValue(val)));
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

export function mapTurnToAnalysisResponse(turn: ConversationTurn): AnalysisResponse {
  const primary = turn.ticker_results?.[0];
  const ticker = turn.tickers?.[0] || primary?.ticker || '';
  const financialData =
    primary?.financial_data ?? primary?.fundamentals_visualization?.raw_data ?? null;
  const fundamentalsVisualization = primary?.fundamentals_visualization ?? null;
  const newsText = turn.agent_analyses?.news_agent || '';
  const fundamentalsText = turn.agent_analyses?.fundamentals_agent || '';
  const synthesis = turn.assistant_synthesis || primary?.analysis_text || '';

  const agents: AgentAnalysis[] = [];

  if (newsText || (primary?.sources?.length ?? 0) > 0) {
    agents.push({
      id: 'news',
      name: 'News Analysis Agent',
      icon: 'news',
      category: 'Intelligence Unit',
      recentCatalyst: {
        title: firstSentence(newsText || synthesis),
        description: (newsText || synthesis).slice(0, 140),
        timeAgo: 'HISTORY',
      },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      fullReport: newsText || synthesis,
      references: (primary?.sources || []).map((s) => ({
        id: s.source_id,
        title: s.title,
        url: s.url,
        source: extractDomain(s.url),
      })),
    });
  }

  if (fundamentalsText || financialData) {
    agents.push({
      id: 'fundamental',
      name: 'Fundamental Agent',
      icon: 'analytics',
      category: 'Financial Lab',
      recentCatalyst: { title: '', description: '', timeAgo: '' },
      sentiment: { score: 50, label: 'NEUTRAL (50%)' },
      metrics: buildMetrics(financialData),
      quote: firstSentence(fundamentalsText || synthesis),
      fullReport: fundamentalsText || synthesis,
      tableData: buildTable(financialData),
    });
  }

  return {
    ticker,
    companyName: ticker || '',
    currentPrice: null,
    priceChange: null,
    priceChangePercent: null,
    marketStatus: 'MARKET DATA UNAVAILABLE',
    chartData: primary?.market_chart ?? [],
    fundamentalData: financialData,
    fundamentalsVisualization,
    agents,
    summary: {
      coreNarrative: synthesis,
      agentConsensus: [
        ...(turn.agent_analyses?.news_agent
          ? [
              {
                title: 'News Sentiment',
                description: 'Recent media coverage and event catalysts assessed.',
                icon: 'verified',
              },
            ]
          : []),
        ...(turn.agent_analyses?.fundamentals_agent
          ? [
              {
                title: 'Fundamental Strength',
                description: 'Financial statements and quantitative metrics evaluated.',
                icon: 'account_balance',
              },
            ]
          : []),
      ],
      verdict: {
        label: 'NEUTRAL',
        description: firstSentence(synthesis || primary?.analysis_text || ''),
      },
    },
  };
}
