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

export type PortfolioAssetType = 'equity' | 'etf';

export interface PortfolioHolding {
  ticker: string;
  company_name: string;
  exchange?: string | null;
  asset_type: PortfolioAssetType;
  shares: number;
}

export interface PortfolioResponse {
  user_email: string;
  holdings: PortfolioHolding[];
}

export interface UpsertPortfolioHoldingRequest {
  user_email?: string | null;
  ticker: string;
  company_name: string;
  exchange?: string | null;
  asset_type: PortfolioAssetType;
  shares: number;
}

export interface AuthRequest {
  email: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
  user_email: string;
  session_id: string;
}

export interface ConversationBootstrapResponse {
  status: 'ok';
  conversation_count: number;
}

export interface TickerSearchResult {
  ticker: string;
  company_name: string;
  exchange?: string | null;
  asset_type: PortfolioAssetType;
}

export interface TickerSearchResponse {
  results: TickerSearchResult[];
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
  fundamentalData?: DataFramePayload | null;
  fundamentalsVisualization?: FundamentalsVisualizationPayload | null;
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
 * The stream emits `progress`, `init`, `chart`, `metrics`, `analysis_chunk`,
 * `complete`, or `error`.
 * We map incremental payloads into the AnalysisResponse shape used by the UI.
 */

export interface DataFramePayload {
  index: string[];
  columns: string[];
  data: Array<Array<number | null>>;
  row_semantics?: Record<
    string,
    {
      value_kind?: string;
      display_unit?: string;
      invalid?: boolean;
      invalid_reason?: string;
    }
  >;
}

export type FundamentalsChartType =
  | 'line'
  | 'bar'
  | 'area'
  | 'scatter'
  | 'stacked_bar'
  | 'stacked_area'
  | 'pie';

export type FundamentalsDataMode = 'timeseries' | 'snapshot';

export interface FundamentalsChartSpecPayload {
  chart_type: FundamentalsChartType;
  data_mode: FundamentalsDataMode;
  snapshot_period: string;
  title: string;
  row_labels: string[];
  group_rows: boolean;
  rationale: string;
}

export interface FundamentalsVisualizationPayload {
  charts: FundamentalsChartSpecPayload[];
  raw_row_labels: string[];
  raw_data?: DataFramePayload | null;
  reviewer_notes: string;
  task_completed: boolean;
  task_completion_reason: string;
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
  market_quote?: MarketQuote | null;
  market_chart?: ChartDataPoint[];
  financial_data?: DataFramePayload | null;
  fundamentals_visualization?: FundamentalsVisualizationPayload | null;
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

export interface ChatAck {
  request_id: string;
  conversation_id: string;
  session_id: string;
}

export interface ConversationSummary {
  conversation_id: string;
  created_at: string;
  last_message_at: string;
  message_count: number;
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ConversationHistoryResponse {
  conversation_id: string;
  messages: ConversationMessage[];
}

export interface ConversationTurn {
  turn_id: string;
  request_id: string;
  conversation_id: string;
  user_email: string;
  session_id: string;
  created_at: string;
  duration_ms: number;
  user_message: string;
  assistant_synthesis: string;
  agent_analyses: Record<string, string>;
  agent_memory_summaries?: Record<string, Record<string, unknown>>;
  ticker_results: TickerResult[];
  tickers: string[];
}

export interface ConversationTurnsResponse {
  conversation_id: string;
  turns: ConversationTurn[];
  has_more: boolean;
  next_before_turn_id: string | null;
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
      event_type: 'fundamentals_visualization';
      request_id: string;
      fundamentals_visualization: FundamentalsVisualizationPayload;
    }
  | {
      event_type: 'ticker_resolved';
      request_id: string;
      ticker?: string;
      tickers?: string[];
    }
  | {
      event_type: 'analysis_chunk';
      request_id: string;
      agent: 'news_agent' | 'fundamentals_agent' | 'orchestrator';
      node: string;
      stream_id: string;
      phase: 'start' | 'delta' | 'end' | 'error';
      seq: number;
      delta?: string;
      text?: string;
      is_final?: boolean;
      error?: string;
    };
