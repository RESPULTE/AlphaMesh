import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  X,
  User,
  Sparkles,
  Clock,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  FileText,
  BarChart2,
  ArrowUp,
} from 'lucide-react';
import Markdown from 'react-markdown';
import {
  Area,
  AreaChart as RechartsAreaChart,
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
  Line,
  LineChart as RechartsLineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {
  ConversationTurn,
  DataFramePayload,
  FundamentalsVisualizationPayload,
  SourceItem,
} from '../types/api';
import {
  getNormalisedFundamentalsCharts,
  normaliseChartSpec,
  toSnapshotDataset,
  toTimeseriesDataset,
} from './charting/fundamentalsChartUtils';
import { groupSourcesByArticle } from '../utils/sourceGrouping';

interface FullConversationViewProps {
  turns: ConversationTurn[];
  conversationId: string;
  onClose: () => void;
  /** Called when user wants to continue in main dashboard view */
  onContinue: () => void;
  isStreaming?: boolean;
  /** Show a loading spinner in the thread area while turns are being fetched */
  isLoading?: boolean;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
}

const CHART_COLORS = ['#007a01', '#2b9f30', '#6ecf72', '#87d98a', '#b6e8b9', '#d1f2d3'];

function formatTimestamp(value: string): string {
  if (!value) return '';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDuration(ms: number): string {
  if (!ms || ms < 0) return '';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ─── Inline chart renderer (subset of AnalysisDashboard logic) ────────────────

function FundamentalsChart({
  financialData,
  visualization,
}: {
  financialData: DataFramePayload | null | undefined;
  visualization: FundamentalsVisualizationPayload | null | undefined;
}) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const charts = getNormalisedFundamentalsCharts(visualization);
  if (!charts.length || !financialData) return null;

  useEffect(() => {
    if (selectedIndex < charts.length) return;
    setSelectedIndex(0);
  }, [charts.length, selectedIndex]);

  const rawSpec = charts[selectedIndex] ?? charts[0];
  const spec = normaliseChartSpec(rawSpec);

  const renderChart = () => {
    if (spec.data_mode === 'snapshot') {
      const { points } = toSnapshotDataset(financialData, spec.row_labels, spec.snapshot_period);
      if (!points.length) return <div className="flex items-center justify-center h-full text-xs text-on-surface-variant/50">No data</div>;
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="name" tick={{ fill: '#888', fontSize: 10 }} />
            <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
            <Tooltip contentStyle={{ background: '#1a1c1c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }} />
            <Bar dataKey="value" fill="#007a01" radius={[4, 4, 0, 0]} />
          </RechartsBarChart>
        </ResponsiveContainer>
      );
    }

    const { points, series } = toTimeseriesDataset(financialData, spec.row_labels);
    if (!points.length || !series.length) return <div className="flex items-center justify-center h-full text-xs text-on-surface-variant/50">No data</div>;

    const isArea = spec.chart_type === 'area' || spec.chart_type === 'stacked_area';
    const isBar = spec.chart_type === 'bar' || spec.chart_type === 'stacked_bar';

    if (isArea) {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsAreaChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="period" tick={{ fill: '#888', fontSize: 10 }} />
            <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
            <Tooltip contentStyle={{ background: '#1a1c1c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {series.map((s, i) => (
              <Area key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={CHART_COLORS[i % CHART_COLORS.length]} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.15} strokeWidth={2} stackId={spec.chart_type === 'stacked_area' ? 'stack' : undefined} />
            ))}
          </RechartsAreaChart>
        </ResponsiveContainer>
      );
    }

    if (isBar) {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RechartsBarChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="period" tick={{ fill: '#888', fontSize: 10 }} />
            <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
            <Tooltip contentStyle={{ background: '#1a1c1c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {series.map((s, i) => (
              <Bar key={s.key} dataKey={s.key} name={s.label} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} stackId={spec.chart_type === 'stacked_bar' ? 'stack' : undefined} />
            ))}
          </RechartsBarChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="period" tick={{ fill: '#888', fontSize: 10 }} />
          <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
          <Tooltip contentStyle={{ background: '#1a1c1c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {series.map((s, i) => (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={CHART_COLORS[i % CHART_COLORS.length]} strokeWidth={2.5} dot={false} />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="mt-4 rounded-xl border border-outline-variant/15 bg-surface overflow-hidden">
      {/* Chart selector tabs */}
      {charts.length > 1 && (
        <div className="flex overflow-x-auto border-b border-outline-variant/10 px-3 pt-2 gap-1">
          {charts.map((c, i) => (
            <button
              key={i}
              onClick={() => setSelectedIndex(i)}
              className={`shrink-0 px-3 py-1.5 rounded-t-lg text-[10px] font-bold font-label tracking-wide transition-colors ${
                selectedIndex === i
                  ? 'bg-primary/10 text-primary border-b-2 border-primary'
                  : 'text-on-surface-variant/60 hover:text-on-surface-variant'
              }`}
            >
              {c.title || `Chart ${i + 1}`}
            </button>
          ))}
        </div>
      )}
      <div className="px-3 py-2 text-[10px] font-bold text-on-surface-variant/60 uppercase tracking-widest font-label">
        {charts[selectedIndex]?.title || 'Financial Chart'}
      </div>
      <div className="h-[200px] px-3 pb-4">{renderChart()}</div>
    </div>
  );
}

// ─── Sources list ─────────────────────────────────────────────────────────────

function SourcesList({ sources }: { sources: SourceItem[] }) {
  if (!sources.length) return null;
  const groupedSources = groupSourcesByArticle(sources);
  const itemClassName =
    'flex items-start gap-2.5 p-2.5 rounded-lg bg-surface-container-lowest border border-outline-variant/10 hover:border-primary/30 hover:bg-surface-container-low/50 transition-all group';
  return (
    <div className="mt-3 space-y-1.5">
      {groupedSources.map((src) =>
        src.url ? (
          <a
            key={src.key}
            href={src.url}
            target="_blank"
            rel="noopener noreferrer"
            className={itemClassName}
          >
            <div className="shrink-0 min-w-5 h-5 rounded-full bg-surface-container-high flex items-center justify-center text-[9px] font-bold text-on-surface-variant group-hover:text-primary transition-colors mt-0.5 px-1">
              {src.citationIds[0] ?? '?'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-on-surface group-hover:text-primary transition-colors line-clamp-2 leading-snug">
                {src.title}
              </p>
              {src.citationIds.length > 0 && (
                <p className="text-[10px] text-on-surface-variant/70 mt-1 font-mono">
                  {src.citationIds.map((id) => `[${id}]`).join(' ')}
                </p>
              )}
            </div>
            <ExternalLink className="w-3 h-3 text-on-surface-variant/30 group-hover:text-primary transition-colors shrink-0 mt-0.5" />
          </a>
        ) : (
          <div key={src.key} className={itemClassName}>
            <div className="shrink-0 min-w-5 h-5 rounded-full bg-surface-container-high flex items-center justify-center text-[9px] font-bold text-on-surface-variant mt-0.5 px-1">
              {src.citationIds[0] ?? '?'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-on-surface line-clamp-2 leading-snug">
                {src.title}
              </p>
              {src.citationIds.length > 0 && (
                <p className="text-[10px] text-on-surface-variant/70 mt-1 font-mono">
                  {src.citationIds.map((id) => `[${id}]`).join(' ')}
                </p>
              )}
            </div>
          </div>
        )
      )}
    </div>
  );
}

// ─── Agent detail panel (collapsible) ────────────────────────────────────────

function AgentDetailPanel({ agentKey, analysisText }: { agentKey: string; analysisText: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-outline-variant/15 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-surface-container hover:bg-surface-container-high transition-colors"
      >
        <div className="flex items-center gap-2.5">
          {agentKey.includes('news') ? (
            <FileText className="w-3.5 h-3.5 text-primary" />
          ) : (
            <BarChart2 className="w-3.5 h-3.5 text-primary" />
          )}
          <span className="text-[10px] font-black font-label tracking-wider uppercase text-on-surface">
            {agentKey.replace(/_/g, ' ')}
          </span>
        </div>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-on-surface-variant/60" /> : <ChevronDown className="w-3.5 h-3.5 text-on-surface-variant/60" />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 py-3 bg-surface-container-lowest border-t border-outline-variant/10">
              <div className="prose prose-sm max-w-none prose-p:text-on-surface-variant prose-p:text-xs prose-p:leading-relaxed prose-li:text-on-surface-variant prose-li:text-xs prose-headings:text-on-surface prose-headings:font-bold prose-strong:text-on-surface prose-a:text-primary">
                <Markdown>{analysisText}</Markdown>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Rich Turn renderer ───────────────────────────────────────────────────────

function RichTurn({ turn, index }: { turn: ConversationTurn; index: number }) {
  const [showAgents, setShowAgents] = useState(false);
  const agentKeys = Object.keys(turn.agent_analyses || {});
  const hasAgents = agentKeys.length > 0;

  // Gather all ticker results with financial data
  const tickerResultsWithCharts = (turn.ticker_results || []).filter(
    (tr) => tr.financial_data && tr.fundamentals_visualization
  );
  const allSources = (turn.ticker_results || []).flatMap((tr) => tr.sources || []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index * 0.05, 0.4) }}
      className="space-y-4"
    >
      {/* User bubble */}
      <div className="flex items-end justify-end gap-2.5">
        <div className="flex flex-col items-end gap-1 max-w-[80%]">
          <div className="bg-primary/15 border border-primary/20 rounded-2xl rounded-br-md px-4 py-3 shadow-sm">
            <p className="text-sm font-medium text-on-surface leading-relaxed">{turn.user_message}</p>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-on-surface-variant/50 font-medium pr-1">
            <Clock className="w-3 h-3" />
            <span>{formatTimestamp(turn.created_at)}</span>
            {turn.duration_ms > 0 && (
              <>
                <span className="opacity-40">·</span>
                <span className="font-mono">{formatDuration(turn.duration_ms)}</span>
              </>
            )}
          </div>
        </div>
        <div className="w-9 h-9 rounded-full bg-surface-container-high border border-outline-variant/20 flex items-center justify-center shrink-0 shadow-sm">
          <User className="w-4 h-4 text-on-surface-variant" />
        </div>
      </div>

      {/* AI bubble */}
      {turn.assistant_synthesis && (
        <div className="flex items-start gap-2.5">
          <div className="w-9 h-9 rounded-full bg-primary/10 border border-primary/15 flex items-center justify-center shrink-0 shadow-sm">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>

          <div className="flex flex-col gap-2.5 flex-1 min-w-0">
            {/* Ticker badges */}
            {turn.tickers && turn.tickers.filter((t) => t?.trim()).length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {turn.tickers.filter((t) => t?.trim()).map((ticker) => (
                  <span key={ticker} className="inline-flex items-center gap-1 bg-surface-container-low border border-outline-variant/20 px-2 py-0.5 rounded-full text-[9px] font-black font-label tracking-wider text-primary uppercase">
                    <TrendingUp className="w-2.5 h-2.5" />{ticker}
                  </span>
                ))}
              </div>
            )}

            {/* Main synthesis */}
            <div className="bg-surface-container border border-outline-variant/15 rounded-2xl rounded-bl-md px-4 py-4 shadow-sm">
              <div className="prose prose-sm max-w-none prose-headings:font-headline prose-headings:font-bold prose-headings:text-on-surface prose-p:text-on-surface-variant prose-p:leading-relaxed prose-li:text-on-surface-variant prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-code:text-primary prose-code:bg-surface-container-highest prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs">
                <Markdown>{turn.assistant_synthesis}</Markdown>
              </div>
            </div>

            {/* Charts from ticker results */}
            {tickerResultsWithCharts.map((tr) => (
              <FundamentalsChart
                key={tr.ticker}
                financialData={tr.financial_data}
                visualization={tr.fundamentals_visualization}
              />
            ))}

            {/* Sources */}
            {allSources.length > 0 && (
              <div className="px-1">
                <div className="text-[9px] font-black font-label tracking-widest text-on-surface-variant/50 uppercase mb-1.5">Sources</div>
                <SourcesList sources={allSources} />
              </div>
            )}

            {/* Agent details toggle */}
            {hasAgents && (
              <div className="px-1 space-y-1.5">
                <button
                  onClick={() => setShowAgents(!showAgents)}
                  className="flex items-center gap-1.5 text-[10px] font-bold font-label tracking-wider uppercase text-on-surface-variant/60 hover:text-on-surface-variant transition-colors"
                >
                  {showAgents ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  {showAgents ? 'Hide' : 'Show'} agent details ({agentKeys.length})
                </button>
                <AnimatePresence>
                  {showAgents && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden space-y-1.5"
                    >
                      {agentKeys.map((key) => (
                        <AgentDetailPanel
                          key={key}
                          agentKey={key}
                          analysisText={turn.agent_analyses[key] || ''}
                        />
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}

// ─── Main modal ──────────────────────────────────────────────────────────────

export default function FullConversationView({
  turns,
  conversationId,
  onClose,
  onContinue,
  isStreaming = false,
  isLoading = false,
  hasMore = false,
  isLoadingMore = false,
  onLoadMore,
}: FullConversationViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const hasInitialBottomScrollRef = useRef(false);
  const pendingPrependOffsetRef = useRef<{ scrollTop: number; scrollHeight: number } | null>(null);
  const requestedMoreRef = useRef(false);

  // Reset per-conversation transient state.
  useEffect(() => {
    hasInitialBottomScrollRef.current = false;
    pendingPrependOffsetRef.current = null;
    requestedMoreRef.current = false;
  }, [conversationId]);

  // Prevent body scroll while modal is open.
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  // Keep "latest first" robust: once initial turns load, jump to bottom exactly once.
  useEffect(() => {
    if (isLoading || turns.length === 0 || hasInitialBottomScrollRef.current) return;
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' });
      hasInitialBottomScrollRef.current = true;
    });
  }, [isLoading, turns.length]);

  // Preserve viewport position when older turns are prepended.
  useEffect(() => {
    if (!pendingPrependOffsetRef.current) return;
    if (isLoadingMore) return;
    const thread = threadRef.current;
    if (!thread) return;
    const { scrollTop, scrollHeight } = pendingPrependOffsetRef.current;
    const delta = thread.scrollHeight - scrollHeight;
    thread.scrollTop = Math.max(0, scrollTop + delta);
    pendingPrependOffsetRef.current = null;
  }, [isLoadingMore, turns.length]);

  useEffect(() => {
    if (!isLoadingMore) {
      requestedMoreRef.current = false;
    }
  }, [isLoadingMore]);

  const handleThreadScroll = useCallback(() => {
    const thread = threadRef.current;
    if (!thread) return;
    if (!onLoadMore || !hasMore || isLoading || isLoadingMore || requestedMoreRef.current) return;

    const scrollableDelta = thread.scrollHeight - thread.clientHeight;
    // Only fetch older turns when the list is genuinely scrollable and user reaches near-top.
    if (scrollableDelta <= 24) return;
    if (thread.scrollTop > 72) return;

    requestedMoreRef.current = true;
    pendingPrependOffsetRef.current = {
      scrollTop: thread.scrollTop,
      scrollHeight: thread.scrollHeight,
    };
    onLoadMore();
  }, [hasMore, isLoading, isLoadingMore, onLoadMore]);

  const shortId = `${conversationId.slice(0, 8)}...${conversationId.slice(-6)}`;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/70 backdrop-blur-md"
        onClick={onClose}
      />

      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.97 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="relative flex flex-col w-full h-[100dvh] md:h-[88vh] md:max-w-3xl md:rounded-3xl overflow-hidden bg-surface shadow-2xl border border-outline-variant/15 z-10"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="shrink-0 flex items-center justify-between px-5 py-4 border-b border-outline-variant/10 bg-surface-container-lowest">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/15 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-primary" />
            </div>
            <div>
              <div className="text-[9px] font-black font-label tracking-widest text-on-surface-variant/50 uppercase">
                AlphaMesh Conversation
              </div>
              <div className="text-sm font-bold text-on-surface font-mono">{shortId}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden md:block text-[10px] font-medium text-on-surface-variant/50">
              {turns.length} turn{turns.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-full hover:bg-surface-container-high flex items-center justify-center text-on-surface-variant transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div
          ref={threadRef}
          onScroll={handleThreadScroll}
          className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 pb-32 space-y-8 bg-surface-container-lowest/30"
        >
          {isLoadingMore && (
            <div className="flex items-center justify-center py-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                className="w-5 h-5 border-2 border-primary/20 border-t-primary rounded-full"
              />
            </div>
          )}

          {isLoading ? (
            <div className="flex flex-col items-center justify-center gap-4 py-24">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                className="w-8 h-8 border-2 border-primary/20 border-t-primary rounded-full"
              />
              <p className="text-sm text-on-surface-variant/60 font-medium">Loading conversation...</p>
            </div>
          ) : turns.length === 0 ? (
            <div className="flex items-center justify-center py-24 text-sm text-on-surface-variant/50">
              No messages in this conversation yet.
            </div>
          ) : (
            turns.map((turn, i) => <RichTurn key={turn.turn_id} turn={turn} index={i} />)
          )}
          <div ref={bottomRef} className="h-2" />
        </div>

        <div className="absolute bottom-0 left-0 right-0 border-t border-outline-variant/10 px-4 py-5 bg-transparent relative overflow-hidden">
          {/* Soft tint so content remains visible behind the action strip */}
          <div className="absolute inset-0 bg-surface-container-lowest/18 pointer-events-none" />
          {/* Gradient depth: lighter at top, denser at bottom */}
          <div className="absolute inset-0 bg-gradient-to-t from-surface-container-lowest/72 via-surface-container-lowest/36 to-transparent pointer-events-none" />
          {/* Backdrop blur gradient layers */}
          <div className="absolute inset-x-0 top-0 h-10 backdrop-blur-[1px] pointer-events-none" />
          <div className="absolute inset-x-0 top-8 h-14 backdrop-blur-[3px] pointer-events-none" />
          <div className="absolute inset-x-0 bottom-0 h-24 backdrop-blur-xl pointer-events-none" />
          <div className="relative flex items-center justify-center">
            <button
              type="button"
              onClick={onContinue}
              disabled={isStreaming}
              aria-label="Continue conversation in dashboard"
              className="group relative w-14 h-14 md:w-16 md:h-16 rotate-45 bg-primary/20 border border-primary/40 backdrop-blur-xl rounded-2xl flex items-center justify-center hover:bg-primary/30 hover:shadow-[0_0_24px_rgba(0,200,5,0.35)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="-rotate-45">
                <ArrowUp className="w-5 h-5 md:w-6 md:h-6 text-primary group-hover:-translate-y-0.5 transition-transform" />
              </div>
            </button>
          </div>
          <p className="text-[9px] text-center text-on-surface-variant/40 mt-3 font-black uppercase tracking-[0.25em]">
            Continue in dashboard
          </p>
        </div>
      </motion.div>
    </div>
  );
}
