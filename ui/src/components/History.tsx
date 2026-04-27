import { motion, AnimatePresence } from 'motion/react';
import { MessageSquare } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { ConversationSummary, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import AnalysisDashboard from './AnalysisDashboard';
import FullConversationView from './FullConversationView';

interface HistoryProps {
  query?: string | null;
  onClearQuery?: () => void;
  /** When set, auto-open this conversation on mount */
  initialExpandedId?: string | null;
  /** Called when user submits a new message inside the full-view modal */
  onContinueConversation?: (conversationId: string, query: string) => void;
}

const DEV_USER_EMAIL = 'demo@alphamesh.local';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';

function formatTimestamp(value: string): string {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function truncateLabel(text: string, max = 72): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

export default function History({
  query,
  onClearQuery,
  initialExpandedId,
  onContinueConversation,
}: HistoryProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationTurns, setConversationTurns] = useState<Record<string, ConversationTurn[]>>({});
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [loadingTurnsFor, setLoadingTurnsFor] = useState<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_CONVERSATION_ID) : null
  );
  /** Which conversation is open in the full-screen modal */
  const [fullViewId, setFullViewId] = useState<string | null>(null);

  // Track whether initialExpandedId has already triggered an open
  const initialOpenDone = useRef(false);

  // ── Load conversation list ───────────────────────────────────────────────
  useEffect(() => {
    if (query) return;
    let cancelled = false;

    const load = async () => {
      setIsLoadingConversations(true);
      try {
        const res = await fetch(
          `/api/v1/conversations?limit=50&user_email=${encodeURIComponent(DEV_USER_EMAIL)}`
        );
        if (!res.ok || cancelled) return;
        const rows = (await res.json()) as ConversationSummary[];
        if (!cancelled) setConversations(Array.isArray(rows) ? rows : []);
      } finally {
        if (!cancelled) setIsLoadingConversations(false);
      }
    };

    load();
    return () => { cancelled = true; };
  }, [query]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setActiveConversationId(window.localStorage.getItem(STORAGE_CONVERSATION_ID));
  }, [query]);

  // ── Auto-open a specific conversation (e.g. from Chat tab deep-link) ────
  useEffect(() => {
    if (initialExpandedId && !initialOpenDone.current) {
      initialOpenDone.current = true;
      openFullView(initialExpandedId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialExpandedId, conversations]);

  // ── Lazily load turns for a conversation ────────────────────────────────
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
        [conversationId]: payload.turns || [],
      }));
    } finally {
      setLoadingTurnsFor(null);
    }
  };

  // ── Open the full-screen modal ───────────────────────────────────────────
  const openFullView = async (conversationId: string) => {
    await loadTurns(conversationId);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_CONVERSATION_ID, conversationId);
    }
    setActiveConversationId(conversationId);
    setFullViewId(conversationId);
  };

  const closeFullView = () => setFullViewId(null);

  const handleContinue = (newQuery: string) => {
    if (fullViewId && onContinueConversation) {
      onContinueConversation(fullViewId, newQuery);
    }
    closeFullView();
  };

  // ── If a live analysis query is active, show the dashboard ──────────────
  if (query) {
    return <AnalysisDashboard query={query} onBack={onClearQuery} />;
  }

  const fullViewTurns = fullViewId ? (conversationTurns[fullViewId] ?? []) : [];

  return (
    <>
      {/* ── Full-screen conversation modal ──────────────────────────── */}
      <AnimatePresence>
        {fullViewId && (
          <FullConversationView
            turns={fullViewTurns}
            conversationId={fullViewId}
            onClose={closeFullView}
            onContinue={handleContinue}
            isLoading={loadingTurnsFor === fullViewId}
          />
        )}
      </AnimatePresence>

      {/* ── Page ────────────────────────────────────────────────────── */}
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

        {isLoadingConversations && (
          <div className="flex items-center gap-3 text-sm text-on-surface-variant">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
              className="w-4 h-4 border-2 border-primary/20 border-t-primary rounded-full"
            />
            Loading conversation history…
          </div>
        )}

        {!isLoadingConversations && conversations.length === 0 && (
          <div className="text-sm text-on-surface-variant">
            No conversation history yet. Run an analysis to create your first session.
          </div>
        )}

        {/* ── Conversation card list (static, click-to-open) ────────── */}
        <div className="space-y-3 md:space-y-4">
          {conversations.map((conv, i) => {
            const isActive = activeConversationId === conv.conversation_id;
            // Use the cached first-turn user message as label if available
            const cachedTurns = conversationTurns[conv.conversation_id];
            const label = cachedTurns?.[0]?.user_message
              ? truncateLabel(cachedTurns[0].user_message)
              : truncateLabel(conv.conversation_id);

            return (
              <motion.div
                key={conv.conversation_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: i * 0.04 }}
                onClick={() => openFullView(conv.conversation_id)}
                className="group flex items-center gap-4 md:gap-6 p-5 md:p-8 bg-surface-container-lowest rounded-xl cursor-pointer hover:bg-surface-container-low transition-colors border border-transparent hover:border-outline-variant/10 shadow-[0_4px_20px_rgba(0,0,0,0.05)]"
              >
                {/* Icon */}
                <div className="w-12 h-12 md:w-14 md:h-14 bg-surface-container rounded-full flex items-center justify-center shrink-0 group-hover:bg-primary/10 transition-colors">
                  <MessageSquare className="w-5 h-5 md:w-6 md:h-6 text-primary" />
                </div>

                {/* Text */}
                <div className="flex-1 min-w-0">
                  <h2 className="font-headline text-base md:text-lg font-bold tracking-tight text-on-surface truncate group-hover:text-primary transition-colors">
                    {label}
                  </h2>
                  <p className="text-outline text-xs md:text-sm mt-0.5">
                    {conv.message_count} messages · Last active {formatTimestamp(conv.last_message_at)}
                  </p>
                  {isActive && (
                    <p className="text-primary text-xs font-semibold mt-1">Active conversation</p>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.main>
    </>
  );
}
