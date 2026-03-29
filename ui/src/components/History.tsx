import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, ChevronRight, MessageSquare, Download, FileText } from 'lucide-react';
import { useState } from 'react';
import AnalysisDashboard from './AnalysisDashboard';

interface HistoryProps {
  onAnalyze: (query: string) => void;
  query?: string | null;
  onClearQuery?: () => void;
}

export default function History({ onAnalyze, query, onClearQuery }: HistoryProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const history = [
    {
      id: 'AAPL',
      name: 'Apple Inc.',
      sessions: '3 Recent Sessions',
      lastAnalyzed: '2h ago',
      logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuC2KoNz-C0qjcLpR9Adu58LvMA6b_U_6GRFHA-y-yvEMbjMeQlJ6lZ1NIzLBp6aV8ae8mgfcEv9xqGZqp6UucztF3uzG6mdQ55oJmJfmVcwIHnMzZJZdkFOGPaLTIWhLXSMEo9nJ3DnGV1lfTsJWAuCy1KjPxH-wpGVpLNp2eSGn30Jyd3kHf2p6T6NjFyHlT3we-oHV8xw1yZiZ1L6MZGluEdQrumUKhDfSYD0txlFMHaFlxU_Q_do0f-QQwiVbh-ANCrniYx86Mk',
      data: {
        pe: '28.4x',
        div: '0.52%',
        margin: '26.3%',
        rsi: '54.2',
        summary: 'The core thesis focuses on Services segment growth offsetting hardware cycles. Key focus in recent chat was the potential integration of proprietary generative models into the ecosystem. Sentiment remains cautious but constructive on margin expansion.'
      },
      pastSessions: [
        { id: 's1', name: 'Q3 Earnings Deep Dive', date: '2h ago' },
        { id: 's2', name: 'Vision Pro Supply Chain Impact', date: 'Oct 15, 2023' },
        { id: 's3', name: 'Services Revenue Growth Model', date: 'Oct 10, 2023' },
        { id: 's4', name: 'Hardware Cycle Analysis', date: 'Sep 28, 2023' },
        { id: 's5', name: 'Generative AI Integration', date: 'Sep 15, 2023' },
      ]
    },
    {
      id: 'TSLA',
      name: 'Tesla, Inc.',
      sessions: '1 Recent Session',
      lastAnalyzed: 'Oct 12, 2023',
      logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuChVzVsmmbzHjWR9VhE1sP8OXb4KHBRCq1Ep5-iMoZp_srfj75nhO1KOy6N0EOx-A0NdGkLgN8snxEfeTZh_PIBd8yZIT6W9eOzSLA_VMP-Yf9N9eGNIAiRtqewkfc3TOfY2jEunTx2r9Crx4lgNEhBJ5Ttp5NgcNu5y81-OgYwwi-Ds64_nQiivIqR8jStrgl5oYx0GLB-rXh6hJJes-fkExWPYC-yHFOw7X0E4ZZijHWCYbvlAYlmQZVk1MCkTm11J98lyfzHmI0',
      pastSessions: [
        { id: 's6', name: 'Margin Compression Analysis', date: 'Oct 12, 2023' },
      ]
    },
    {
      id: 'NVDA',
      name: 'NVIDIA Corp',
      sessions: '5 Recent Sessions',
      lastAnalyzed: 'yesterday',
      logo: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCPmDbUiqAqAzAvVfVtXjC0eryWS6Lu8yf5H2cr73586QJ6vxqmwOrOt9c9wGcYetNf1ZbDkj7JlMEv9SU8An0So9APODBYF8tpEJznJY3ZQL0hlivzva8P9lza42uDF952OUzQeJqekKCFUYv1r3oAF9rAh2o2e1YjAi3mQ1iRsVSZs5J996cHW9FoOX2dWCaY9MbMYfg15plllqEAF5oAv07P_MOFFBcT1cAk5Hhfic3WEsFxZVU4FBrknXpU7u1tmv7EF0AIk_k',
      pastSessions: [
        { id: 's7', name: 'Data Center Revenue Forecast', date: 'yesterday' },
        { id: 's8', name: 'H100 Demand Metrics', date: 'Oct 18, 2023' },
        { id: 's9', name: 'Competitor Landscape (AMD)', date: 'Oct 15, 2023' },
        { id: 's10', name: 'Supply Constraints Review', date: 'Oct 10, 2023' },
        { id: 's11', name: 'Gaming Segment Recovery', date: 'Oct 5, 2023' },
      ]
    }
  ];

  if (query) {
    return <AnalysisDashboard query={query} onBack={onClearQuery} />;
  }

  return (
    <motion.main
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="pt-24 md:pt-28 pb-24 md:pb-32 px-4 md:px-6 max-w-7xl mx-auto w-full"
    >
      <div className="mb-10 md:mb-16">
        <span className="font-label text-[10px] md:text-[0.6875rem] uppercase tracking-widest text-outline mb-1.5 md:mb-2 block">
          AlphaMesh History
        </span>
        <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tight text-on-surface">
          Analysis History
        </h1>
        <p className="text-sm md:text-base text-on-surface-variant mt-3 md:mt-4 max-w-lg leading-relaxed">
          Review your deep-dive sessions, captured metrics, and architectural summaries of company performance over time.
        </p>
      </div>

      <div className="space-y-4 md:space-y-6">
        {history.map((item) => (
          <section
            key={item.id}
            className="bg-surface-container-lowest rounded-xl overflow-hidden shadow-[0_20px_40px_rgba(26,28,28,0.06)] group"
          >
            <div
              onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
              className="p-5 md:p-8 flex items-center justify-between cursor-pointer hover:bg-surface-container-low transition-colors"
            >
              <div className="flex items-center gap-4 md:gap-6">
                <div className="w-12 h-12 md:w-16 md:h-16 bg-surface-container rounded-full flex items-center justify-center shrink-0">
                  <img src={item.logo} alt={item.name} className="w-6 h-6 md:w-10 md:h-10 object-contain grayscale" referrerPolicy="no-referrer" />
                </div>
                <div>
                  <h2 className="font-headline text-xl md:text-2xl font-bold tracking-tight text-on-surface">{item.name}</h2>
                  <p className="text-outline text-xs md:text-sm">{item.sessions} • Last analyzed {item.lastAnalyzed}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 md:gap-4 shrink-0">
                <ChevronDown 
                  className={`w-5 h-5 md:w-6 md:h-6 text-outline transition-transform duration-300 ${
                    expandedId === item.id ? 'rotate-180' : ''
                  }`} 
                />
              </div>
            </div>

            <AnimatePresence>
              {expandedId === item.id && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: 'easeInOut' }}
                  className="border-t border-outline-variant/10"
                >
                  <div className="p-5 md:p-8 bg-surface-container-lowest/50">
                    <div className="mb-6">
                      <h3 className="text-xs font-bold tracking-widest text-on-surface-variant/60 uppercase mb-4 font-label">
                        Session History
                      </h3>
                      <div className="bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10">
                        <div className="max-h-[220px] overflow-y-auto custom-scrollbar">
                          {item.pastSessions?.map((session, index) => (
                            <div 
                              key={session.id}
                              className={`flex items-center justify-between p-4 hover:bg-surface-container-high transition-colors ${
                                index !== (item.pastSessions?.length || 0) - 1 ? 'border-b border-outline-variant/10' : ''
                              }`}
                            >
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                                  <MessageSquare className="w-4 h-4 text-primary" />
                                </div>
                                <div>
                                  <h4 className="font-medium text-on-surface text-sm">{session.name}</h4>
                                  <p className="text-xs text-on-surface-variant mt-0.5">{session.date}</p>
                                </div>
                              </div>
                              <button 
                                onClick={() => onAnalyze(`${item.name}: ${session.name}`)}
                                className="flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors px-3 py-1.5 rounded-lg hover:bg-primary/10"
                              >
                                View in Chat
                                <ChevronRight className="w-4 h-4" />
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {item.data && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-surface-container p-4 rounded-xl">
                          <div className="text-xs text-on-surface-variant mb-1">P/E Ratio</div>
                          <div className="font-mono text-lg text-on-surface">{item.data.pe}</div>
                        </div>
                        <div className="bg-surface-container p-4 rounded-xl">
                          <div className="text-xs text-on-surface-variant mb-1">Div Yield</div>
                          <div className="font-mono text-lg text-on-surface">{item.data.div}</div>
                        </div>
                        <div className="bg-surface-container p-4 rounded-xl">
                          <div className="text-xs text-on-surface-variant mb-1">Op Margin</div>
                          <div className="font-mono text-lg text-on-surface">{item.data.margin}</div>
                        </div>
                        <div className="bg-surface-container p-4 rounded-xl">
                          <div className="text-xs text-on-surface-variant mb-1">RSI (14d)</div>
                          <div className="font-mono text-lg text-on-surface">{item.data.rsi}</div>
                        </div>
                      </div>
                    )}

                    <div className="flex items-center gap-3">
                      <button className="flex-1 bg-surface-container hover:bg-surface-container-high text-on-surface py-3 rounded-xl text-sm font-medium transition-colors flex items-center justify-center gap-2">
                        <Download className="w-4 h-4" />
                        Export Data
                      </button>
                      <button className="flex-1 bg-primary text-on-primary py-3 rounded-xl text-sm font-medium transition-colors flex items-center justify-center gap-2 hover:bg-primary/90 shadow-lg shadow-primary/20">
                        <FileText className="w-4 h-4" />
                        Full Report
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        ))}
      </div>

      <div className="pt-24 flex flex-col items-center text-center opacity-40">
        <div className="w-32 h-[1px] bg-outline mb-8"></div>
        <p className="font-label text-[0.6875rem] uppercase tracking-[0.2em] text-outline">
          End of History
        </p>
      </div>
    </motion.main>
  );
}
