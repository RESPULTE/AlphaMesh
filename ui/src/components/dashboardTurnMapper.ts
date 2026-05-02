import type {
  AgentAnalysis,
  AnalysisResponse,
  ConversationTurn,
} from '../types/api';
import {
  buildFundamentalsMetrics,
  buildFundamentalsTable,
} from '../lib/fundamentalsDisplay';

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

export function mapTurnToAnalysisResponse(turn: ConversationTurn): AnalysisResponse {
  const primary = turn.ticker_results?.[0];
  const marketQuote = primary?.market_quote;
  const ticker = turn.tickers?.[0] || primary?.ticker || marketQuote?.ticker || '';
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
      metrics: buildFundamentalsMetrics(financialData),
      quote: firstSentence(fundamentalsText || synthesis),
      fullReport: fundamentalsText || synthesis,
      tableData: buildFundamentalsTable(financialData),
    });
  }

  return {
    ticker,
    companyName: marketQuote?.companyName || ticker || '',
    currentPrice: marketQuote?.currentPrice ?? null,
    priceChange: marketQuote?.priceChange ?? null,
    priceChangePercent: marketQuote?.priceChangePercent ?? null,
    marketStatus: marketQuote?.marketStatus || 'MARKET DATA UNAVAILABLE',
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
