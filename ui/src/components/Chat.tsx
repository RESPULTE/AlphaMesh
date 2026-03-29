import React from 'react';
import { motion } from 'motion/react';
import { Search, Sparkles, MessageSquare, ChevronRight } from 'lucide-react';

interface PortfolioProps {
  onAnalyze: (query: string) => void;
}

export default function Chat({ onAnalyze }: PortfolioProps) {
  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
      onAnalyze(e.currentTarget.value.trim());
    }
  };

  const recentSessions = [
    { id: 1, name: 'Apple Q3 Earnings Analysis', date: '2 hours ago' },
    { id: 2, name: 'Tesla Margin Impact', date: 'Yesterday' },
    { id: 3, name: 'NVIDIA AI Chip Demand', date: 'Oct 12, 2023' },
    { id: 4, name: 'Macro: Fed Rate Hike Scenarios', date: 'Oct 10, 2023' },
    { id: 5, name: 'SaaS Multiples Contraction', date: 'Oct 8, 2023' },
  ];

  return (
    <motion.main
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="flex-1 flex flex-col items-center justify-center w-full max-w-7xl mx-auto px-4 md:px-12 pt-28 pb-0 md:pt-24 md:pb-24"
    >
      <section className="w-full max-w-5xl flex flex-col justify-center">
        <div className="mb-8 md:mb-12 text-center">
          <div className="text-[10px] md:text-xs font-bold tracking-[0.2em] text-on-surface-variant/50 uppercase mb-3 md:mb-4 font-label">
            AlphaMesh AI Chat
          </div>
          <h1 className="text-4xl md:text-7xl font-extrabold font-headline tracking-tighter text-on-surface mb-4">
            How can I help you today?
          </h1>
        </div>

        <div className="relative group w-full max-w-3xl mx-auto mb-12">
          <div className="absolute inset-0 bg-primary/5 rounded-2xl md:rounded-3xl blur-xl md:blur-2xl group-focus-within:bg-primary/10 transition-all"></div>
          <div className="relative bg-surface-container-lowest border border-outline-variant/20 rounded-2xl md:rounded-3xl p-1.5 md:p-2 flex items-center shadow-xl shadow-on-surface/5">
            <div className="pl-4 md:pl-6 pr-2 md:pr-4">
              <Search className="w-5 h-5 md:w-6 md:h-6 text-on-surface-variant" />
            </div>
            <input
              type="text"
              placeholder="Search markets or ask AlphaMesh AI..."
              className="w-full bg-transparent border-none focus:ring-0 text-base md:text-xl font-medium py-3 md:py-5 placeholder:text-on-surface-variant/30 outline-none"
              onKeyDown={handleSearch}
            />
            <div className="pr-2 md:pr-4">
              <button className="bg-surface-container-high hover:bg-surface-container-highest p-2 md:p-3 rounded-xl md:rounded-2xl transition-all">
                <Sparkles className="w-5 h-5 md:w-6 md:h-6 text-on-surface-variant" />
              </button>
            </div>
          </div>
        </div>

        <div className="w-full max-w-3xl mx-auto">
          <h3 className="text-sm font-bold tracking-widest text-on-surface-variant/60 uppercase mb-4 font-label px-2">
            Recent Sessions
          </h3>
          <div className="bg-surface-container-lowest border border-outline-variant/20 rounded-2xl overflow-hidden shadow-sm">
            <div className="max-h-[250px] overflow-y-auto">
              {recentSessions.map((session, index) => (
                <div 
                  key={session.id}
                  className={`flex items-center justify-between p-4 hover:bg-surface-container-low transition-colors cursor-pointer ${
                    index !== recentSessions.length - 1 ? 'border-b border-outline-variant/10' : ''
                  }`}
                  onClick={() => onAnalyze(session.name)}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                      <MessageSquare className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-on-surface text-sm md:text-base">{session.name}</h4>
                      <p className="text-xs text-on-surface-variant mt-0.5">{session.date}</p>
                    </div>
                  </div>
                  <button className="flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors px-3 py-1.5 rounded-lg hover:bg-primary/10">
                    View
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </motion.main>
  );
}
