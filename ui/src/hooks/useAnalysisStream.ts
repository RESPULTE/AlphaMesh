import { useState, useEffect } from 'react';
import { AnalysisResponse } from '../types/api';

// Mock streaming implementation
export function useAnalysisStream(query: string | null) {
  const [data, setData] = useState<Partial<AnalysisResponse> | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!query) return;

    setIsStreaming(true);
    setData(null);

    let isMounted = true;

    const runStream = async () => {
      // 1. Initial Data
      if (!isMounted) return;
      setData({
        ticker: 'AAPL',
        companyName: 'Apple Inc.',
        currentPrice: 192.53,
        priceChange: 2.41,
        priceChangePercent: 1.27,
        marketStatus: 'LIVE MARKET OPEN',
        chartData: [],
        agents: [],
        summary: {
          coreNarrative: '',
          agentConsensus: [],
          verdict: { label: '', description: '' }
        }
      });

      await new Promise(r => setTimeout(r, 800));

      // 2. Chart Data
      if (!isMounted) return;
      const mockChart = Array.from({ length: 20 }, (_, i) => ({
        time: `${10 + Math.floor(i / 2)}:${i % 2 === 0 ? '00' : '30'}`,
        price: 185 + Math.random() * 15 + Math.sin(i / 3) * 5
      }));
      setData(prev => prev ? { ...prev, chartData: mockChart } : null);

      await new Promise(r => setTimeout(r, 1000));

      // 3. Agent 1
      if (!isMounted) return;
      setData(prev => prev ? {
        ...prev,
        agents: [
          {
            id: 'news',
            name: 'News Analysis Agent',
            icon: 'news',
            category: 'Intelligence Unit',
            recentCatalyst: {
              title: 'Vision Pro Expansion',
              description: 'Global rollout strategy suggests high confidence in production yields...',
              timeAgo: '2H AGO'
            },
            sentiment: {
              score: 82,
              label: 'BULLISH (82%)'
            },
            fullReport: `## News Analysis Report: Apple Inc. (AAPL)

**Executive Summary:**
The intelligence unit has detected a significant positive shift in media sentiment over the past 48 hours, primarily driven by the accelerated global rollout of the Vision Pro headset [1].

### Key Catalysts
* **Vision Pro Expansion:** Supply chain sources indicate production yields have improved by 15%, allowing for an earlier-than-expected launch in European and Asian markets [2].
* **Services Revenue:** App Store developer payouts suggest a record-breaking quarter for the Services division, offsetting any potential hardware softness [3].
* **AI Integration Rumors:** Speculation is mounting regarding a major generative AI announcement at the upcoming WWDC, with several key AI researchers recently joining the company [4].

### Sentiment Breakdown
* **Retail Sentiment:** Highly Bullish (88/100)
* **Institutional Sentiment:** Cautiously Optimistic (75/100)
* **Media Tone:** Positive (82/100)

**Conclusion:** The narrative is shifting from "hardware saturation" to "ecosystem expansion and AI integration," providing a strong tailwind for the stock in the near term.`,
            references: [
              { id: 1, title: 'Apple Accelerates Vision Pro Rollout', url: 'https://example.com/news/1', source: 'TechCrunch' },
              { id: 2, title: 'Supply Chain Yields Improve for Mixed Reality Headsets', url: 'https://example.com/news/2', source: 'Bloomberg' },
              { id: 3, title: 'App Store Revenue Hits New Highs', url: 'https://example.com/news/3', source: 'CNBC' },
              { id: 4, title: 'Apple Poaches Key AI Talent Ahead of WWDC', url: 'https://example.com/news/4', source: 'The Verge' }
            ]
          }
        ]
      } : null);

      await new Promise(r => setTimeout(r, 1000));

      // 4. Agent 2
      if (!isMounted) return;
      setData(prev => prev ? {
        ...prev,
        agents: [
          ...(prev.agents || []),
          {
            id: 'fundamental',
            name: 'Fundamental Agent',
            icon: 'analytics',
            category: 'Financial Lab',
            recentCatalyst: {
              title: '',
              description: '',
              timeAgo: ''
            },
            sentiment: { score: 0, label: '' },
            metrics: [
              { label: 'P/E RATIO', value: '28.4x' },
              { label: 'ROE', value: '145%' }
            ],
            quote: '"Net margins expanded by 40bps due to service-mix shift."',
            fullReport: `## Fundamental Analysis: Apple Inc. (AAPL)

**Financial Health Overview:**
Apple's balance sheet remains a fortress. The company continues to generate massive free cash flow, allowing for aggressive share repurchases and consistent dividend growth.

### Key Metrics Analysis
* **Valuation (P/E 28.4x):** While trading at a premium to the broader market, the valuation is justified by the increasing share of high-margin Services revenue.
* **Profitability (ROE 145%):** Exceptional return on equity demonstrates management's efficiency in capital allocation.
* **Margins:** Gross margins have expanded to 45.2%, up 40 basis points year-over-year, driven by a favorable product mix and lower component costs.

### Risk Factors
1. **Geopolitical Exposure:** Reliance on overseas manufacturing remains a tail risk, though diversification efforts in India and Vietnam are accelerating.
2. **Consumer Spending:** A prolonged macroeconomic downturn could impact upgrade cycles for flagship devices.

**Verdict:** The fundamental picture is exceptionally strong. The transition towards a recurring revenue model (Services) provides a floor for valuation multiples, making AAPL a core holding for long-term capital appreciation.`,
            tableData: {
              title: "Financial Metrics Comparison",
              headers: ["Metric", "AAPL", "MSFT", "GOOGL"],
              rows: [
                ["P/E Ratio", "28.4x", "35.2x", "24.1x"],
                ["P/S Ratio", "7.5x", "12.8x", "6.2x"],
                ["ROE", "145%", "38%", "26%"],
                ["Gross Margin", "45.2%", "69.8%", "56.5%"],
                ["Operating Margin", "30.1%", "43.2%", "28.4%"],
                ["Free Cash Flow Yield", "3.8%", "2.5%", "4.1%"]
              ]
            }
          }
        ]
      } : null);

      await new Promise(r => setTimeout(r, 800));

      // 5. Stream Summary
      const fullNarrative = "Apple is successfully transitioning from a hardware-reliant model to a diversified ecosystem player. The recent expansion of Vision Pro signals a maturing AR/VR strategy that markets are beginning to price in more favorably.";
      
      for (let i = 0; i <= fullNarrative.length; i++) {
        if (!isMounted) return;
        setData(prev => {
          if (!prev || !prev.summary) return prev;
          return {
            ...prev,
            summary: {
              ...prev.summary,
              coreNarrative: fullNarrative.substring(0, i)
            }
          };
        });
        await new Promise(r => setTimeout(r, 20));
      }

      await new Promise(r => setTimeout(r, 500));

      // 6. Final Consensus & Verdict
      if (!isMounted) return;
      setData(prev => {
        if (!prev || !prev.summary) return prev;
        return {
          ...prev,
          summary: {
            ...prev.summary,
            agentConsensus: [
              {
                title: 'News Sentiment (82%)',
                description: 'Public perception is shifting towards a growth-oriented AI future rather than defensive stagnation.',
                icon: 'verified'
              },
              {
                title: 'Fundamentals (Solid)',
                description: 'Exceptional ROE and margin expansion provide a significant safety buffer for new R&D investments.',
                icon: 'account_balance'
              }
            ],
            verdict: {
              label: 'STRONG BUY',
              description: 'Cross-agent verification suggests a high probability of short-term alpha relative to SPY benchmarks.'
            }
          }
        };
      });

      setIsStreaming(false);
    };

    runStream();

    return () => {
      isMounted = false;
    };
  }, [query]);

  return { data, isStreaming };
}
