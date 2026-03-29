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

export interface AnalysisResponse {
  ticker: string;
  companyName: string;
  currentPrice: number;
  priceChange: number;
  priceChangePercent: number;
  marketStatus: string;
  chartData: ChartDataPoint[];
  agents: AgentAnalysis[];
  summary: SummaryOfFindings;
}

/**
 * API Integration Guidelines for Python Backend
 * 
 * To integrate this frontend with your Python backend, you should implement a 
 * Server-Sent Events (SSE) or WebSocket endpoint that streams the `AnalysisResponse` 
 * object in chunks.
 * 
 * Recommended Approach (Server-Sent Events):
 * 
 * 1. Endpoint: `GET /api/analyze?query={user_prompt}`
 * 2. Response Headers: `Content-Type: text/event-stream`
 * 3. Event Stream Format:
 * 
 * The backend should yield partial JSON objects as they are generated. 
 * For example, you can stream the `coreNarrative` character by character, 
 * or stream each `AgentAnalysis` as it completes.
 * 
 * Example Python (FastAPI) Backend:
 * ```python
 * from fastapi import FastAPI
 * from fastapi.responses import StreamingResponse
 * import json
 * import asyncio
 * 
 * app = FastAPI()
 * 
 * async def generate_analysis(query: str):
 *     # 1. Send initial shell (ticker, price, empty arrays)
 *     yield f"data: {json.dumps({'type': 'init', 'data': {'ticker': 'AAPL', 'companyName': 'Apple Inc.', 'currentPrice': 192.53}})}\n\n"
 *     
 *     # 2. Stream chart data
 *     await asyncio.sleep(1)
 *     yield f"data: {json.dumps({'type': 'chart', 'data': [{'time': '10:00', 'price': 190}, ...]})}\n\n"
 *     
 *     # 3. Stream agent 1
 *     await asyncio.sleep(2)
 *     yield f"data: {json.dumps({'type': 'agent', 'data': {'id': 'news', 'name': 'News Analysis Agent', ...}})}\n\n"
 *     
 *     # 4. Stream summary text chunk by chunk
 *     summary_text = "Apple is successfully transitioning..."
 *     for i in range(len(summary_text)):
 *         yield f"data: {json.dumps({'type': 'summary_chunk', 'data': summary_text[i]})}\n\n"
 *         await asyncio.sleep(0.05)
 *         
 *     yield f"data: {json.dumps({'type': 'done'})}\n\n"
 * 
 * @app.get("/api/analyze")
 * async def analyze(query: str):
 *     return StreamingResponse(generate_analysis(query), media_type="text/event-stream")
 * ```
 * 
 * In the frontend, we use an `EventSource` to listen to these events and 
 * update the React state incrementally, creating a fluid loading experience.
 */
