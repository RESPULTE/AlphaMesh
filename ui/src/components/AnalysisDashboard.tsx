import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { ArrowLeft } from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import { useAnalysisStream } from '../hooks/useAnalysisStream';
import type { AnalysisResponse, ConversationTurn, ConversationTurnsResponse } from '../types/api';
import DashboardTurnPanel from './DashboardTurnPanel';
import { mapTurnToAnalysisResponse } from './dashboardTurnMapper';

interface AnalysisDashboardProps {
  query?: string | null;
  queryVersion?: number;
  conversationIdOverride?: string | null;
  onBack?: () => void;
  onStreamingChange?: (isStreaming: boolean) => void;
}

const HISTORY_PAGE_SIZE = 8;
const LIVE_PANEL_OFFSET_TOP = 96;
const FAR_SCROLL_DISTANCE = 900;
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';

const EMPTY_LIVE_RESPONSE: AnalysisResponse = {
  ticker: '',
  companyName: '',
  currentPrice: null,
  priceChange: null,
  priceChangePercent: null,
  marketStatus: 'MARKET DATA UNAVAILABLE',
  chartData: [],
  fundamentalData: null,
  fundamentalsVisualization: null,
  agents: [],
  summary: {
    coreNarrative: '',
    agentConsensus: [],
    verdict: { label: '', description: '' },
  },
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

function cubicBezierProgress(
  t: number,
  p1x: number,
  p1y: number,
  p2x: number,
  p2y: number
): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;

  const cx = 3 * p1x;
  const bx = 3 * (p2x - p1x) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * p1y;
  const by = 3 * (p2y - p1y) - cy;
  const ay = 1 - cy - by;

  const sampleCurveX = (u: number) => ((ax * u + bx) * u + cx) * u;
  const sampleCurveY = (u: number) => ((ay * u + by) * u + cy) * u;
  const sampleCurveDerivativeX = (u: number) => (3 * ax * u + 2 * bx) * u + cx;

  let u = t;
  for (let i = 0; i < 6; i++) {
    const x = sampleCurveX(u) - t;
    const dx = sampleCurveDerivativeX(u);
    if (Math.abs(dx) < 1e-6) break;
    u -= x / dx;
    u = clamp(u, 0, 1);
  }

  return sampleCurveY(u);
}

export default function AnalysisDashboard({
  query,
  queryVersion = 0,
  conversationIdOverride,
  onBack,
  onStreamingChange,
}: AnalysisDashboardProps) {
  const { authFetch } = useAuth();
  const {
    data,
    isStreaming,
    streamPhase,
    conversationId: streamedConversationId,
    requestId,
  } = useAnalysisStream(
    query ?? null,
    queryVersion
  );
  const conversationId = conversationIdOverride ?? streamedConversationId;
  const [historicalTurns, setHistoricalTurns] = useState<ConversationTurn[]>([]);
  const [hasMoreHistory, setHasMoreHistory] = useState(false);
  const [nextBeforeTurnId, setNextBeforeTurnId] = useState<string | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [historyLoadedForConversation, setHistoryLoadedForConversation] = useState<string | null>(
    null
  );
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const liveAnchorRef = useRef<HTMLDivElement>(null);
  const prevStreamingRef = useRef(false);
  const autoScrolledConversationRef = useRef<string | null>(null);
  const scrollAnimationRef = useRef<number | null>(null);
  const lastAutoScrollQueryVersionRef = useRef<number | null>(null);
  const staleRecoveryRunForConversationRef = useRef<string | null>(null);

  const recoverFromStaleConversation = useCallback(async () => {
    if (!conversationId) return;
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(STORAGE_CONVERSATION_ID);
    }
    setHistoricalTurns([]);
    setHasMoreHistory(false);
    setNextBeforeTurnId(null);
    setHistoryLoadedForConversation(conversationId);

    if (staleRecoveryRunForConversationRef.current === conversationId) return;
    staleRecoveryRunForConversationRef.current = conversationId;
    try {
      await authFetch('/api/v1/conversations?limit=1');
    } catch {
      // best-effort only
    }
  }, [authFetch, conversationId]);

  const loadHistoryPage = useCallback(
    async (opts?: { reset?: boolean; beforeTurnId?: string | null }) => {
      if (!conversationId || isLoadingHistory) return;
      const reset = opts?.reset ?? false;

      setIsLoadingHistory(true);
      try {
        const params = new URLSearchParams({
          limit: String(HISTORY_PAGE_SIZE),
        });
        if (!reset && opts?.beforeTurnId) {
          params.set('before_turn_id', opts.beforeTurnId);
        }

        const res = await authFetch(
          `/api/v1/conversations/${encodeURIComponent(conversationId)}/turns?${params.toString()}`
        );
        if (res.status === 404) {
          await recoverFromStaleConversation();
          return;
        }
        if (!res.ok) {
          if (reset) {
            setHistoryLoadedForConversation(conversationId);
          }
          return;
        }

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
    [authFetch, conversationId, isLoadingHistory, recoverFromStaleConversation]
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
      // Delay slightly so the backend has time to commit the turn before we re-fetch.
      const timer = setTimeout(() => void loadHistoryPage({ reset: true }), 800);
      return () => clearTimeout(timer);
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

  const hasLiveData = Boolean(data);
  const hasActiveStreamQuery = Boolean(query);
  const isStreamFinished = streamPhase === 'completed' || streamPhase === 'error';
  const shouldHideLiveTurn = !hasActiveStreamQuery || (isStreamFinished && liveTurnAlreadyPersisted);

  useEffect(() => {
    onStreamingChange?.(isStreaming);
  }, [isStreaming, onStreamingChange]);

  useEffect(() => {
    return () => {
      if (scrollAnimationRef.current != null) {
        cancelAnimationFrame(scrollAnimationRef.current);
      }
      onStreamingChange?.(false);
    };
  }, [onStreamingChange]);

  const smoothScrollTo = useCallback((targetY: number) => {
    const startY = window.scrollY;
    const distance = Math.abs(targetY - startY);
    if (distance < 4) return;

    if (scrollAnimationRef.current != null) {
      cancelAnimationFrame(scrollAnimationRef.current);
    }

    const duration = clamp(980 - 0.28 * distance, 560, 980);
    const startTime = performance.now();
    const bezier =
      distance >= FAR_SCROLL_DISTANCE
        ? ([0.2, 0.7, 0.1, 1] as const)
        : ([0.22, 0.61, 0.36, 1] as const);

    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = cubicBezierProgress(progress, bezier[0], bezier[1], bezier[2], bezier[3]);
      const nextY = startY + (targetY - startY) * eased;
      window.scrollTo({ top: nextY, behavior: 'auto' });
      if (progress < 1) {
        scrollAnimationRef.current = requestAnimationFrame(step);
      } else {
        scrollAnimationRef.current = null;
      }
    };

    scrollAnimationRef.current = requestAnimationFrame(step);
  }, []);

  useEffect(() => {
    if (!hasActiveStreamQuery) return;
    if (lastAutoScrollQueryVersionRef.current === queryVersion) return;
    const targetEl = liveAnchorRef.current;
    if (!targetEl) return;
    lastAutoScrollQueryVersionRef.current = queryVersion;

    // Delay measuring until after the full layout flush — a plain RAF fires before
    // Framer Motion's layout pass completes, giving a stale rect.
    const timer = setTimeout(() => {
      if (!liveAnchorRef.current) return;
      const rect = liveAnchorRef.current.getBoundingClientRect();
      const targetTop = Math.max(0, window.scrollY + rect.top - LIVE_PANEL_OFFSET_TOP);
      const distance = Math.abs(targetTop - window.scrollY);

      // If the anchor is already close to the current viewport bottom, let Framer
      // Motion's `layout` prop animate the gentle push-up — no explicit scroll needed.
      if (distance <= 250) return;

      smoothScrollTo(targetTop);
    }, 100);

    return () => clearTimeout(timer);
  }, [hasActiveStreamQuery, queryVersion, smoothScrollTo]);

  useEffect(() => {
    if (hasActiveStreamQuery) return;
    if (!conversationId || historyLoadedForConversation !== conversationId) return;
    if (autoScrolledConversationRef.current === conversationId) return;
    if (!hasLiveData && historicalTurns.length === 0) return;

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
    hasActiveStreamQuery,
    hasLiveData,
    historicalTurns.length,
  ]);

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
        {historicalTurns.map((turn) => (
          <div key={turn.turn_id}>
            <DashboardTurnPanel
              query={turn.user_message}
              data={mapTurnToAnalysisResponse(turn)}
              isStreaming={false}
              streamPhase="completed"
            />
          </div>
        ))}

        {hasActiveStreamQuery && <div ref={liveAnchorRef} className="h-1" />}

        <AnimatePresence initial={false}>
          {!shouldHideLiveTurn && (
            <motion.div
              key={`live-${queryVersion}`}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15, ease: 'easeOut' }}
            >
              <DashboardTurnPanel
                query={query ?? ''}
                data={(data as AnalysisResponse) ?? EMPTY_LIVE_RESPONSE}
                isStreaming={isStreaming}
                streamPhase={streamPhase}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.main>
  );
}
