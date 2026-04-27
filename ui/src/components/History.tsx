import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, ChevronRight, MessageSquare } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { ConversationSummary, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import AnalysisDashboard from './AnalysisDashboard';

interface HistoryProps {
  query?: string | null;
  onClearQuery?: () => void;
}

const DEV_USER_EMAIL = 'demo@alphamesh.local';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';

function formatTimestamp(value: string): string {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function truncateConversationId(conversationId: string): string {
  if (conversationId.length <= 18) return conversationId;
  return `${conversationId.slice(0, 8)}...${conversationId.slice(-6)}`;
}

export default function History({ query, onClearQuery }: HistoryProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationTurns, setConversationTurns] = useState<Record<string, ConversationTurn[]>>({});
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [loadingTurnsFor, setLoadingTurnsFor] = useState<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_CONVERSATION_ID) : null
  );

  useEffect(() => {
    if (query) return;
    let cancelled = false;

    const loadConversations = async () => {
      setIsLoadingConversations(true);
      try {
        const res = await fetch(
          `/api/v1/conversations?limit=50&user_email=${encodeURIComponent(DEV_USER_EMAIL)}`
        );
        if (!res.ok) return;
        const rows = (await res.json()) as ConversationSummary[];
        if (cancelled) return;
        setConversations(Array.isArray(rows) ? rows : []);
      } finally {
        if (!cancelled) setIsLoadingConversations(false);
      }
    };

    loadConversations();
    return () => {
      cancelled = true;
    };
  }, [query]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setActiveConversationId(window.localStorage.getItem(STORAGE_CONVERSATION_ID));
  }, [query]);

  const loadTurns = async (conversationId: string) => {
    if (conversationTurns[conversationId]) return;
    setLoadingTurnsFor(conversationId);
    try {
      const res = await fetch(
        `/api/v1/conversations/${encodeURIComponent(conversationId)}/turns?user_email=${encodeURIComponent(DEV_USER_EMAIL)}`
      );
      if (!res.ok) return;
      const payload = (await res.json()) as ConversationTurnsResponse;
      setConversationTurns((prev) => ({
        ...prev,
        [conversationId]: payload.turns || []
      }));
    } finally {
      setLoadingTurnsFor(null);
    }
  };

  const handleToggle = async (conversationId: string) => {
    const nextExpanded = expandedId === conversationId ? null : conversationId;
    setExpandedId(nextExpanded);
    if (nextExpanded) {
      await loadTurns(nextExpanded);
    }
  };

  const continueConversation = (conversationId: string) => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_CONVERSATION_ID, conversationId);
    }
    setActiveConversationId(conversationId);
  };

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
      </div>

      {isLoadingConversations ? (
        <div className="text-sm text-on-surface-variant">Loading conversation history...</div>
      ) : null}

      {!isLoadingConversations && conversations.length === 0 ? (
        <div className="text-sm text-on-surface-variant">
          No conversation history yet. Run an analysis to create your first turn.
        </div>
      ) : null}

      <div className="space-y-4 md:space-y-6">
        {conversations.map((conversation) => {
          const turns = conversationTurns[conversation.conversation_id] || [];
          const latestTurn = turns.length ? turns[turns.length - 1] : null;
          const isExpanded = expandedId === conversation.conversation_id;
          const isActive = activeConversationId === conversation.conversation_id;
          return (
            <section
              key={conversation.conversation_id}
              className="bg-surface-container-lowest rounded-xl overflow-hidden shadow-[0_20px_40px_rgba(26,28,28,0.06)] group"
            >
              <div
                onClick={() => handleToggle(conversation.conversation_id)}
                className="p-5 md:p-8 flex items-center justify-between cursor-pointer hover:bg-surface-container-low transition-colors"
              >
                <div className="flex items-center gap-4 md:gap-6">
                  <div className="w-12 h-12 md:w-16 md:h-16 bg-surface-container rounded-full flex items-center justify-center shrink-0">
                    <MessageSquare className="w-6 h-6 md:w-8 md:h-8 text-primary" />
                  </div>
                  <div>
                    <h2 className="font-headline text-xl md:text-2xl font-bold tracking-tight text-on-surface">
                      {truncateConversationId(conversation.conversation_id)}
                    </h2>
                    <p className="text-outline text-xs md:text-sm">
                      {conversation.message_count} messages • Last analyzed {formatTimestamp(conversation.last_message_at)}
                    </p>
                    {isActive ? (
                      <p className="text-primary text-xs font-semibold mt-1">Active conversation</p>
                    ) : null}
                  </div>
                </div>
                <ChevronDown
                  className={`w-5 h-5 md:w-6 md:h-6 text-outline transition-transform duration-300 ${
                    isExpanded ? 'rotate-180' : ''
                  }`}
                />
              </div>

              <AnimatePresence>
                {isExpanded ? (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25, ease: 'easeInOut' }}
                    className="border-t border-outline-variant/10"
                  >
                    <div className="p-5 md:p-8 bg-surface-container-lowest/50">
                      {loadingTurnsFor === conversation.conversation_id ? (
                        <div className="text-sm text-on-surface-variant mb-4">Loading turns...</div>
                      ) : null}

                      {latestTurn ? (
                        <div className="mb-6 p-4 rounded-xl bg-surface-container border border-outline-variant/10">
                          <h3 className="text-xs font-bold tracking-widest text-on-surface-variant/60 uppercase mb-2 font-label">
                            Latest Synthesis
                          </h3>
                          <p className="text-sm text-on-surface">{latestTurn.assistant_synthesis || 'No synthesis text'}</p>
                        </div>
                      ) : null}

                      <h3 className="text-xs font-bold tracking-widest text-on-surface-variant/60 uppercase mb-4 font-label">
                        Turns
                      </h3>
                      <div className="bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10">
                        <div className="max-h-[280px] overflow-y-auto custom-scrollbar">
                          {turns.length === 0 ? (
                            <div className="p-4 text-sm text-on-surface-variant">No turns found.</div>
                          ) : (
                            turns
                              .slice()
                              .reverse()
                              .map((turn) => (
                                <div
                                  key={turn.turn_id}
                                  className="flex items-center justify-between p-4 border-b border-outline-variant/10 last:border-b-0"
                                >
                                  <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                                      <MessageSquare className="w-4 h-4 text-primary" />
                                    </div>
                                    <div>
                                      <h4 className="font-medium text-on-surface text-sm line-clamp-1">
                                        {turn.user_message}
                                      </h4>
                                      <p className="text-xs text-on-surface-variant mt-0.5">
                                        {formatTimestamp(turn.created_at)}
                                      </p>
                                    </div>
                                  </div>
                                  <button
                                    onClick={() =>
                                      continueConversation(conversation.conversation_id)
                                    }
                                    className="flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors px-3 py-1.5 rounded-lg hover:bg-primary/10"
                                  >
                                    Continue
                                    <ChevronRight className="w-4 h-4" />
                                  </button>
                                </div>
                              ))
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </section>
          );
        })}
      </div>
    </motion.main>
  );
}
