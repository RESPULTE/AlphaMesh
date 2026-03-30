export interface AgentAnalysis {
  id: string;
  name: string;
  icon: string;
  category: string;
  recentCatalyst: {
    title: string;
    description: string;
    timeAgo: string;
  };
  sentiment: {
    score: number; // 0 to 100
    label: string; // e.g., "BULLISH (82%)"
  };
  metrics?: {
    label: string;
    value: string;
  }[];
  quote?: string;
  fullReport?: string;
  tableData?: {
    title: string;
    headers: string[];
    rows: string[][];
  };
  references?: {
    id: number;
    title: string;
    url: string;
    source: string;
  }[];
}

export interface SummaryOfFindings {
  coreNarrative: string;
  agentConsensus: {
    title: string;
    description: string;
    icon: string;
  }[];
  verdict: {
    label: string;
    description: string;
  };
}

export interface ChartDataPoint {
  time: string;
  price: number;
}

export interface MarketQuote {
  ticker: string;
  companyName: string;
  currentPrice: number;
  priceChange: number;
  priceChangePercent: number;
  marketStatus: string;
}

export interface AnalysisResponse {
  ticker: string;
  companyName: string;
  currentPrice: number | null;
  priceChange: number | null;
  priceChangePercent: number | null;
  marketStatus: string;
  chartData: ChartDataPoint[];
  agents: AgentAnalysis[];
  summary: SummaryOfFindings;
}

/**
 * API Integration Notes
 *
 * Current flow:
 * 1) POST /api/v1/chat  � start analysis, returns request_id
 * 2) GET  /api/v1/stream/{request_id}  � SSE stream
 *
 * The stream emits `progress`, `init`, `chart`, `metrics`, `complete`, or `error`.
 * We map incremental payloads into the AnalysisResponse shape used by the UI.
 */

export interface DataFramePayload {
  index: string[];
  columns: string[];
  data: Array<Array<number | null>>;
}

export interface SourceItem {
  source_id: number;
  title: string;
  url: string;
  page_content?: string;
}

export interface TickerResult {
  ticker: string;
  analysis_text: string;
  financial_data?: DataFramePayload | null;
  sources: SourceItem[];
}

export interface FinalResult {
  request_id: string;
  conversation_id: string;
  synthesis: string;
  ticker_results: TickerResult[];
  agent_analyses: Record<string, string>;
  duration_ms: number;
}

export type StreamEvent =
  | {
      event_type: 'progress';
      request_id: string;
      source?: string;
      level?: string;
      message?: string;
      timestamp?: string;
    }
  | {
      event_type: 'complete';
      request_id: string;
      result: FinalResult;
    }
  | {
      event_type: 'error';
      request_id: string;
      error: string;
    }
  | {
      event_type: 'init';
      request_id: string;
      quote: MarketQuote;
    }
  | {
      event_type: 'chart';
      request_id: string;
      chart: ChartDataPoint[];
    }
  | {
      event_type: 'metrics';
      request_id: string;
      financial_data: DataFramePayload;
    }
  | {
      event_type: 'ticker_resolved';
      request_id: string;
      ticker?: string;
      tickers?: string[];
    };
