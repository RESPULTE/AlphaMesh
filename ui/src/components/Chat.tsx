import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Search, Sparkles, MessageSquare, ChevronRight } from 'lucide-react';
import type { ConversationSummary } from '../types/api';
import { useAuth } from '../auth/AuthContext';

interface ChatProps {
  onAnalyze: (query: string) => void;
  /** Called when user wants to open an existing conversation by ID */
  onOpenConversation?: (conversationId: string) => void;
}

function timeSince(isoString: string): string {
  if (!isoString) return '';
  const date = new Date(isoString);
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffMins = Math.floor(diffMs / 60_000);
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHrs = Math.floor(diffMins / 60);
  if (diffHrs < 24) return `${diffHrs}h ago`;
  const diffDays = Math.floor(diffHrs / 24);
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function Chat({ onAnalyze, onOpenConversation }: ChatProps) {
  const { authFetch } = useAuth();
  const [inputValue, setInputValue] = useState('');
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchConversations = async () => {
      setIsLoadingConversations(true);
      try {
        const res = await authFetch('/api/v1/conversations?limit=10');
        if (!res.ok || cancelled) return;
        const rows = (await res.json()) as ConversationSummary[];
        if (!cancelled) setConversations(Array.isArray(rows) ? rows : []);
      } catch {
        // silently ignore fetch errors in the chat landing page
      } finally {
        if (!cancelled) setIsLoadingConversations(false);
      }
    };
    fetchConversations();
    return () => { cancelled = true; };
  }, [authFetch]);

  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
      onAnalyze(e.currentTarget.value.trim());
      setInputValue('');
    }
  };

  const handleSendClick = () => {
    if (inputValue.trim()) {
      onAnalyze(inputValue.trim());
      setInputValue('');
    }
  };

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

        {/* ── Search / input bar ────────────────────────────────────────── */}
        <div className="relative group w-full max-w-3xl mx-auto mb-12">
          <div className="absolute inset-0 bg-primary/5 rounded-2xl md:rounded-3xl blur-xl md:blur-2xl group-focus-within:bg-primary/10 transition-all"></div>
          <div className="relative bg-surface-container-lowest border border-outline-variant/20 rounded-2xl md:rounded-3xl p-1.5 md:p-2 flex items-center shadow-xl shadow-on-surface/5">
            <div className="pl-4 md:pl-6 pr-2 md:pr-4">
              <Search className="w-5 h-5 md:w-6 md:h-6 text-on-surface-variant" />
            </div>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Search markets or ask AlphaMesh AI..."
              className="w-full bg-transparent border-none focus:ring-0 text-base md:text-xl font-medium py-3 md:py-5 placeholder:text-on-surface-variant/30 outline-none"
              onKeyDown={handleSearch}
            />
            <div className="pr-2 md:pr-4">
              <button
                onClick={handleSendClick}
                className="bg-surface-container-high hover:bg-surface-container-highest p-2 md:p-3 rounded-xl md:rounded-2xl transition-all"
              >
                <Sparkles className="w-5 h-5 md:w-6 md:h-6 text-on-surface-variant" />
              </button>
            </div>
          </div>
        </div>

        {/* ── Recent Sessions ──────────────────────────────────────────── */}
        <div className="w-full max-w-3xl mx-auto">
          <div className="flex items-center justify-between mb-4 px-2">
            <h3 className="text-sm font-bold tracking-widest text-on-surface-variant/60 uppercase font-label">
              Recent Sessions
            </h3>
            {conversations.length > 0 && (
              <span className="text-[10px] font-bold text-on-surface-variant/40 uppercase tracking-wider font-label">
                {conversations.length} conversations
              </span>
            )}
          </div>

          <div className="bg-surface-container-lowest border border-outline-variant/20 rounded-2xl overflow-hidden shadow-sm">
            {isLoadingConversations ? (
              <div className="p-6 flex items-center gap-3 text-sm text-on-surface-variant/60">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                  className="w-4 h-4 border-2 border-primary/20 border-t-primary rounded-full"
                />
                Loading recent sessions...
              </div>
            ) : conversations.length === 0 ? (
              <div className="p-6 text-sm text-on-surface-variant/60 text-center">
                No sessions yet — run your first analysis above.
              </div>
            ) : (
              <div className="max-h-[300px] overflow-y-auto custom-scrollbar divide-y divide-outline-variant/10">
                {conversations.map((conv) => (
                  <div
                    key={conv.conversation_id}
                    className="flex items-center justify-between p-4 hover:bg-surface-container-low transition-colors cursor-pointer"
                    onClick={() => onOpenConversation?.(conv.conversation_id)}
                  >
                    <div className="flex items-center gap-4 min-w-0">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                        <MessageSquare className="w-4 h-4 text-primary" />
                      </div>
                      <div className="min-w-0">
                        <h4 className="font-semibold text-on-surface text-sm md:text-base truncate">
                          {conv.conversation_id.slice(0, 8)}…{conv.conversation_id.slice(-6)}
                        </h4>
                        <p className="text-xs text-on-surface-variant mt-0.5">
                          {conv.message_count} messages · {timeSince(conv.last_message_at)}
                        </p>
                      </div>
                    </div>
                    <button className="flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors px-3 py-1.5 rounded-lg hover:bg-primary/10 shrink-0">
                      View
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>
    </motion.main>
  );
}
