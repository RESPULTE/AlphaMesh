import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, MessageSquare, ChevronRight } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { ConversationSummary, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import AnalysisDashboard from './AnalysisDashboard';
import ConversationThread from './ConversationThread';
import ConversationFullView from './ConversationFullView';

interface HistoryProps {
  query?: string | null;
  onClearQuery?: () => void;
  /** When set, auto-expand and load this conversation on mount */
  initialExpandedId?: string | null;
  /** Called when user submits a new message from the full-view chat input */
  onAnalyze?: (query: string) => void;
  isStreaming?: boolean;
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

export default function History({
  query,
  onClearQuery,
  initialExpandedId,
  onAnalyze,
  isStreaming = false,
}: HistoryProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationTurns, setConversationTurns] = useState<Record<string, ConversationTurn[]>>({});
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [loadingTurnsFor, setLoadingTurnsFor] = useState<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_CONVERSATION_ID) : null
  );
  /** When set, show the full-page ConversationFullView overlay */
  const [fullViewId, setFullViewId] = useState<string | null>(null);

  const expandedRef = useRef<HTMLElement | null>(null);

  // ── Load conversation list ───────────────────────────────────────────────
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
    return () => { cancelled = true; };
  }, [query]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setActiveConversationId(window.localStorage.getItem(STORAGE_CONVERSATION_ID));
  }, [query]);

  // ── Auto-expand when deep-linked ─────────────────────────────────────────
  useEffect(() => {
    if (initialExpandedId && initialExpandedId !== expandedId) {
      setExpandedId(initialExpandedId);
      loadTurns(initialExpandedId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialExpandedId]);

  useEffect(() => {
    if (expandedId && expandedRef.current) {
      expandedRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [expandedId, conversations]);

  // ── Fetch turns for a conversation ───────────────────────────────────────
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
    if (nextExpanded) await loadTurns(nextExpanded);
  };

  const openFullView = (conversationId: string) => {
    window.localStorage.setItem(STORAGE_CONVERSATION_ID, conversationId);
    setActiveConversationId(conversationId);
    setFullViewId(conversationId);
  };

  const handleContinueAnalyze = (q: string) => {
    // Close full view and hand off to the normal analysis stream
    setFullViewId(null);
    onAnalyze?.(q);
  };

  // ── Analysis dashboard mode ──────────────────────────────────────────────
  if (query) {
    return <AnalysisDashboard query={query} onBack={onClearQuery} />;
  }

  return (
    <>
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
            const isExpanded = expandedId === conversation.conversation_id;
            const isActive = activeConversationId === conversation.conversation_id;

            const firstTurn = turns.length ? turns[0] : null;
            const displayLabel = firstTurn?.user_message
              ? firstTurn.user_message.length > 72
                ? `${firstTurn.user_message.slice(0, 72)}…`
                : firstTurn.user_message
              : truncateConversationId(conversation.conversation_id);

            return (
              <section
                key={conversation.conversation_id}
                ref={isExpanded ? (el) => { expandedRef.current = el; } : undefined}
                className="bg-surface-container-lowest rounded-xl overflow-hidden shadow-[0_20px_40px_rgba(26,28,28,0.06)] group"
              >
                {/* ── Header ─────────────────────────────────────────── */}
                <div
                  onClick={() => handleToggle(conversation.conversation_id)}
                  className="p-5 md:p-8 flex items-center justify-between cursor-pointer hover:bg-surface-container-low transition-colors"
                >
                  <div className="flex items-center gap-4 md:gap-6 min-w-0">
                    <div className="w-12 h-12 md:w-16 md:h-16 bg-surface-container rounded-full flex items-center justify-center shrink-0">
                      <MessageSquare className="w-6 h-6 md:w-8 md:h-8 text-primary" />
                    </div>
                    <div className="min-w-0">
                      <h2 className="font-headline text-base md:text-xl font-bold tracking-tight text-on-surface line-clamp-1">
                        {displayLabel}
                      </h2>
                      <p className="text-outline text-xs md:text-sm mt-0.5">
                        {conversation.message_count} messages · Last active {formatTimestamp(conversation.last_message_at)}
                      </p>
                      {isActive ? (
                        <p className="text-primary text-xs font-semibold mt-1">Active conversation</p>
                      ) : null}
                    </div>
                  </div>
                  <ChevronDown
                    className={`w-5 h-5 md:w-6 md:h-6 text-outline transition-transform duration-300 shrink-0 ml-4 ${
                      isExpanded ? 'rotate-180' : ''
                    }`}
                  />
                </div>

                {/* ── Preview thread (max 6 turns) ───────────────────── */}
                <AnimatePresence>
                  {isExpanded ? (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: 'easeInOut' }}
                      className="border-t border-outline-variant/10"
                    >
                      {/* Loading */}
                      {loadingTurnsFor === conversation.conversation_id && (
                        <div className="flex items-center justify-center gap-3 py-10 text-sm text-on-surface-variant">
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                            className="w-5 h-5 border-2 border-primary/20 border-t-primary rounded-full"
                          />
                          Loading conversation…
                        </div>
                      )}

                      {!loadingTurnsFor && (
                        <>
                          {turns.length > 1 && (
                            <div className="px-5 md:px-8 pt-5 pb-0">
                              <div className="text-[9px] font-black tracking-widest text-on-surface-variant/50 uppercase mb-2 font-label">
                                {turns.length} turns in this conversation
                              </div>
                            </div>
                          )}

                          {/* Preview thread — capped at 6 turns */}
                          <ConversationThread
                            turns={turns}
                            maxTurns={6}
                            onViewFull={() => openFullView(conversation.conversation_id)}
                          />

                          {/* Footer */}
                          <div className="border-t border-outline-variant/10 px-5 md:px-8 py-4 flex items-center justify-between bg-surface-container-lowest/60">
                            <span className="text-xs text-on-surface-variant/60 font-medium">
                              Conversation {truncateConversationId(conversation.conversation_id)}
                            </span>
                            <button
                              onClick={() => openFullView(conversation.conversation_id)}
                              className="flex items-center gap-1.5 text-xs font-bold text-primary hover:text-primary/80 transition-colors px-4 py-2 rounded-lg hover:bg-primary/10"
                            >
                              Continue conversation
                              <ChevronRight className="w-4 h-4" />
                            </button>
                          </div>
                        </>
                      )}
                    </motion.div>
                  ) : null}
                </AnimatePresence>
              </section>
            );
          })}
        </div>
      </motion.main>

      {/* ── Full-page conversation overlay ─────────────────────────────────── */}
      <AnimatePresence>
        {fullViewId && (
          <ConversationFullView
            key={fullViewId}
            conversationId={fullViewId}
            turns={conversationTurns[fullViewId] || []}
            onBack={() => setFullViewId(null)}
            onAnalyze={handleContinueAnalyze}
            isStreaming={isStreaming}
          />
        )}
      </AnimatePresence>
    </>
  );
}
