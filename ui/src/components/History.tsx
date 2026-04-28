import { motion, AnimatePresence } from 'motion/react';
import { MessageSquare } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { ConversationSummary, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import AnalysisDashboard from './AnalysisDashboard';
import FullConversationView from './FullConversationView';

interface HistoryProps {
  query?: string | null;
  onClearQuery?: () => void;
  dashboardConversationId?: string | null;
  onClearDashboardConversation?: () => void;
  initialExpandedId?: string | null;
  onContinueConversation?: (conversationId: string) => void;
}

const DEV_USER_EMAIL = 'demo@alphamesh.local';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';
const TURN_PAGE_SIZE = 8;

function formatTimestamp(value: string): string {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function truncateLabel(text: string, max = 72): string {
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

export default function History({
  query,
  onClearQuery,
  dashboardConversationId,
  onClearDashboardConversation,
  initialExpandedId,
  onContinueConversation,
}: HistoryProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [loadingTurnsFor, setLoadingTurnsFor] = useState<string | null>(null);
  const [isLoadingMoreTurns, setIsLoadingMoreTurns] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_CONVERSATION_ID) : null
  );
  const [fullViewId, setFullViewId] = useState<string | null>(null);
  const [fullViewTurns, setFullViewTurns] = useState<ConversationTurn[]>([]);
  const [fullViewHasMore, setFullViewHasMore] = useState(false);
  const [fullViewNextBefore, setFullViewNextBefore] = useState<string | null>(null);

  const initialOpenDone = useRef(false);

  useEffect(() => {
    if (query || dashboardConversationId) return;
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

    void load();
    return () => {
      cancelled = true;
    };
  }, [query, dashboardConversationId]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    setActiveConversationId(window.localStorage.getItem(STORAGE_CONVERSATION_ID));
  }, [query, dashboardConversationId]);

  useEffect(() => {
    if (initialExpandedId && !initialOpenDone.current) {
      initialOpenDone.current = true;
      void openFullView(initialExpandedId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialExpandedId, conversations]);

  const loadTurnsPage = async (
    conversationId: string,
    opts?: { reset?: boolean; beforeTurnId?: string | null }
  ) => {
    const reset = opts?.reset ?? false;
    if (reset) {
      setLoadingTurnsFor(conversationId);
    } else {
      setIsLoadingMoreTurns(true);
    }
    try {
      const params = new URLSearchParams({
        user_email: DEV_USER_EMAIL,
        limit: String(TURN_PAGE_SIZE),
      });
      if (!reset && opts?.beforeTurnId) {
        params.set('before_turn_id', opts.beforeTurnId);
      }
      const res = await fetch(
        `/api/v1/conversations/${encodeURIComponent(conversationId)}/turns?${params.toString()}`
      );
      if (!res.ok) return;
      const payload = (await res.json()) as ConversationTurnsResponse;
      const incoming = payload.turns || [];
      setFullViewTurns((prev) => {
        if (reset) return incoming;
        const seen = new Set(prev.map((turn) => turn.turn_id));
        const prepended = incoming.filter((turn) => !seen.has(turn.turn_id));
        return [...prepended, ...prev];
      });
      setFullViewHasMore(Boolean(payload.has_more));
      setFullViewNextBefore(payload.next_before_turn_id ?? null);
    } finally {
      setLoadingTurnsFor(null);
      setIsLoadingMoreTurns(false);
    }
  };

  const openFullView = async (conversationId: string) => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_CONVERSATION_ID, conversationId);
    }
    setActiveConversationId(conversationId);
    setFullViewId(conversationId);
    setFullViewTurns([]);
    setFullViewHasMore(false);
    setFullViewNextBefore(null);
    await loadTurnsPage(conversationId, { reset: true });
  };

  const closeFullView = () => {
    setFullViewId(null);
    setFullViewTurns([]);
    setFullViewHasMore(false);
    setFullViewNextBefore(null);
  };

  const handleContinue = () => {
    if (fullViewId && onContinueConversation) {
      onContinueConversation(fullViewId);
    }
    closeFullView();
  };

  const handleLoadMoreTurns = async () => {
    if (!fullViewId || !fullViewHasMore || !fullViewNextBefore || isLoadingMoreTurns) return;
    await loadTurnsPage(fullViewId, { reset: false, beforeTurnId: fullViewNextBefore });
  };

  if (query || dashboardConversationId) {
    return (
      <AnalysisDashboard
        query={query}
        conversationIdOverride={dashboardConversationId}
        onBack={() => {
          onClearQuery?.();
          onClearDashboardConversation?.();
        }}
      />
    );
  }

  return (
    <>
      <AnimatePresence>
        {fullViewId && (
          <FullConversationView
            turns={fullViewTurns}
            conversationId={fullViewId}
            onClose={closeFullView}
            onContinue={handleContinue}
            isLoading={loadingTurnsFor === fullViewId}
            hasMore={fullViewHasMore}
            isLoadingMore={isLoadingMoreTurns}
            onLoadMore={handleLoadMoreTurns}
          />
        )}
      </AnimatePresence>

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
            Loading conversation history...
          </div>
        )}

        {!isLoadingConversations && conversations.length === 0 && (
          <div className="text-sm text-on-surface-variant">
            No conversation history yet. Run an analysis to create your first session.
          </div>
        )}

        <div className="space-y-3 md:space-y-4">
          {conversations.map((conv, i) => {
            const isActive = activeConversationId === conv.conversation_id;
            const label = truncateLabel(conv.conversation_id);

            return (
              <motion.div
                key={conv.conversation_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: i * 0.04 }}
                onClick={() => void openFullView(conv.conversation_id)}
                className="group flex items-center gap-4 md:gap-6 p-5 md:p-8 bg-surface-container-lowest rounded-xl cursor-pointer hover:bg-surface-container-low transition-colors border border-transparent hover:border-outline-variant/10 shadow-[0_4px_20px_rgba(0,0,0,0.05)]"
              >
                <div className="w-12 h-12 md:w-14 md:h-14 bg-surface-container rounded-full flex items-center justify-center shrink-0 group-hover:bg-primary/10 transition-colors">
                  <MessageSquare className="w-5 h-5 md:w-6 md:h-6 text-primary" />
                </div>

                <div className="flex-1 min-w-0">
                  <h2 className="font-headline text-base md:text-lg font-bold tracking-tight text-on-surface truncate group-hover:text-primary transition-colors">
                    {label}
                  </h2>
                  <p className="text-outline text-xs md:text-sm mt-0.5">
                    {conv.message_count} messages · Last active {formatTimestamp(conv.last_message_at)}
                  </p>
                  {isActive && <p className="text-primary text-xs font-semibold mt-1">Active conversation</p>}
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.main>
    </>
  );
}
