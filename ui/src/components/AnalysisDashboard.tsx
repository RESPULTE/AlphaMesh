import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ArrowLeft } from 'lucide-react';
import { useAnalysisStream } from '../hooks/useAnalysisStream';
import type { AnalysisResponse, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import DashboardTurnPanel from './DashboardTurnPanel';
import { mapTurnToAnalysisResponse } from './dashboardTurnMapper';

interface AnalysisDashboardProps {
  query: string;
  onBack?: () => void;
}

const DEV_USER_EMAIL = 'demo@alphamesh.local';
const HISTORY_PAGE_SIZE = 8;

export default function AnalysisDashboard({ query, onBack }: AnalysisDashboardProps) {
  const { data, isStreaming, conversationId, requestId } = useAnalysisStream(query);
  const [historicalTurns, setHistoricalTurns] = useState<ConversationTurn[]>([]);
  const [hasMoreHistory, setHasMoreHistory] = useState(false);
  const [nextBeforeTurnId, setNextBeforeTurnId] = useState<string | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [historyLoadedForConversation, setHistoryLoadedForConversation] = useState<string | null>(
    null
  );
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const prevStreamingRef = useRef(false);
  const autoScrolledConversationRef = useRef<string | null>(null);

  const loadHistoryPage = useCallback(
    async (opts?: { reset?: boolean; beforeTurnId?: string | null }) => {
      if (!conversationId || isLoadingHistory) return;
      const reset = opts?.reset ?? false;

      setIsLoadingHistory(true);
      try {
        const params = new URLSearchParams({
          user_email: DEV_USER_EMAIL,
          limit: String(HISTORY_PAGE_SIZE),
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

        setHistoricalTurns((prev) => {
          if (reset) return incoming;
          const seen = new Set(prev.map((turn) => turn.turn_id));
          const prepended = incoming.filter((turn) => !seen.has(turn.turn_id));
          return [...prepended, ...prev];
        });
        setHasMoreHistory(Boolean(payload.has_more));
        setNextBeforeTurnId(payload.next_before_turn_id ?? null);
        setHistoryLoadedForConversation(conversationId);
      } finally {
        setIsLoadingHistory(false);
      }
    },
    [conversationId, isLoadingHistory]
  );

  useEffect(() => {
    if (!conversationId) return;
    if (historyLoadedForConversation === conversationId) return;
    setHistoricalTurns([]);
    setHasMoreHistory(false);
    setNextBeforeTurnId(null);
    autoScrolledConversationRef.current = null;
    void loadHistoryPage({ reset: true });
  }, [conversationId, historyLoadedForConversation, loadHistoryPage]);

  useEffect(() => {
    const wasStreaming = prevStreamingRef.current;
    prevStreamingRef.current = isStreaming;
    if (wasStreaming && !isStreaming && conversationId) {
      void loadHistoryPage({ reset: true });
    }
  }, [isStreaming, conversationId, loadHistoryPage]);

  useEffect(() => {
    if (!hasMoreHistory || !nextBeforeTurnId) return;
    const sentinel = topSentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (!entry?.isIntersecting || isLoadingHistory) return;
        void loadHistoryPage({ reset: false, beforeTurnId: nextBeforeTurnId });
      },
      { root: null, threshold: 0.01 }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMoreHistory, nextBeforeTurnId, isLoadingHistory, loadHistoryPage]);

  const liveTurnAlreadyPersisted = useMemo(() => {
    if (!requestId) return false;
    return historicalTurns.some((turn) => turn.request_id === requestId);
  }, [historicalTurns, requestId]);

  const filteredHistoricalTurns = useMemo(() => {
    if (!requestId) return historicalTurns;
    return historicalTurns.filter((turn) => turn.request_id !== requestId);
  }, [historicalTurns, requestId]);

  const hasLiveData = Boolean(data);

  useEffect(() => {
    if (!conversationId || historyLoadedForConversation !== conversationId) return;
    if (autoScrolledConversationRef.current === conversationId) return;
    if (!hasLiveData && filteredHistoricalTurns.length === 0) return;

    requestAnimationFrame(() => {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: 'auto',
      });
      autoScrolledConversationRef.current = conversationId;
    });
  }, [
    conversationId,
    historyLoadedForConversation,
    hasLiveData,
    filteredHistoricalTurns.length,
  ]);

  if (!hasLiveData) {
    return (
      <div className="pt-32 pb-24 px-6 md:px-12 flex flex-col items-center justify-center min-h-screen w-full max-w-[1600px] mx-auto">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
          className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full"
        />
        <p className="mt-4 text-on-surface-variant font-medium">Initializing AlphaMesh Agents...</p>
      </div>
    );
  }

  return (
    <motion.main
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.02 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="pt-24 md:pt-28 pb-32 md:pb-36 px-4 md:px-8 w-full max-w-[1600px] mx-auto flex-grow"
    >
      {onBack && (
        <button
          onClick={onBack}
          className="mb-6 flex items-center gap-2 text-on-surface-variant hover:text-on-surface transition-colors font-medium text-sm md:text-base py-2 px-4 rounded-full hover:bg-surface-container-low w-fit"
        >
          <ArrowLeft className="w-4 h-4 md:w-5 md:h-5" />
          Back to History
        </button>
      )}

      <div ref={topSentinelRef} className="h-2" />

      {hasMoreHistory && isLoadingHistory && (
        <div className="flex justify-center mb-6">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
            className="w-6 h-6 border-2 border-primary/20 border-t-primary rounded-full"
          />
        </div>
      )}

      <div className="space-y-8 md:space-y-10">
        {filteredHistoricalTurns.map((turn) => (
          <DashboardTurnPanel
            key={turn.turn_id}
            query={turn.user_message}
            data={mapTurnToAnalysisResponse(turn)}
            isStreaming={false}
          />
        ))}

        {!liveTurnAlreadyPersisted && (
          <DashboardTurnPanel
            key={`live-${requestId ?? query}`}
            query={query}
            data={data as AnalysisResponse}
            isStreaming={isStreaming}
          />
        )}
      </div>
    </motion.main>
  );
}
