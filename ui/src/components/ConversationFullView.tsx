/**
 * ConversationFullView.tsx
 *
 * Full-page chat replay for a historical conversation.
 * Renders all turns with:
 *  - User bubble
 *  - AI synthesis (Markdown)
 *  - Ticker badges
 *  - Agent analysis (expandable)
 *  - Fundamentals charts (if available)
 *  - Source citations
 *
 * A sticky ChatInput at the bottom lets the user continue the conversation.
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  ArrowLeft,
  User,
  Sparkles,
  Clock,
  TrendingUp,
  FileText,
  BarChart2,
  ChevronDown,
  ExternalLink,
} from 'lucide-react';
import { ArrowUp, Plus } from 'lucide-react';
import Markdown from 'react-markdown';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { ConversationTurn, DataFramePayload, FundamentalsVisualizationPayload } from '../types/api';
import {
  normaliseChartSpec,
  toTimeseriesDataset,
  toSnapshotDataset,
} from './charting/fundamentalsChartUtils';

const CHART_COLORS = ['#007a01', '#2b9f30', '#6ecf72', '#87d98a', '#b6e8b9'];
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';

// ─── helpers ────────────────────────────────────────────────────────────────

function fmtTimestamp(value: string) {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function fmtDuration(ms: number) {
  if (!ms || ms < 0) return '';
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

// ─── Inline chart renderer ───────────────────────────────────────────────────

function TurnChart({
  financialData,
  visualization,
}: {
  financialData: DataFramePayload | null | undefined;
  visualization: FundamentalsVisualizationPayload | null | undefined;
}) {
  const firstSpec = visualization?.charts?.[0];
  if (!firstSpec || !financialData) return null;

  const spec = normaliseChartSpec(firstSpec);

  if (spec.data_mode === 'snapshot') {
    const { points } = toSnapshotDataset(financialData, spec.row_labels, spec.snapshot_period);
    if (!points.length) return null;
    return (
      <div className="mt-4 bg-surface/60 rounded-2xl p-4 border border-outline-variant/10">
        <p className="text-[9px] font-black font-label tracking-widest text-on-surface-variant/50 uppercase mb-3">
          {spec.title || 'Financial Snapshot'}
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2a2c2b" />
            <XAxis dataKey="name" tick={{ fill: '#888', fontSize: 10 }} />
            <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
            <Tooltip contentStyle={{ background: '#1a1c1b', border: '1px solid #2a2c2b', borderRadius: 8 }} />
            <Bar dataKey="value" fill={CHART_COLORS[0]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // timeseries
  const { points, series } = toTimeseriesDataset(financialData, spec.row_labels);
  if (!points.length || !series.length) return null;

  const ChartComp = spec.chart_type === 'area' || spec.chart_type === 'stacked_area' ? AreaChart : LineChart;

  return (
    <div className="mt-4 bg-surface/60 rounded-2xl p-4 border border-outline-variant/10">
      <p className="text-[9px] font-black font-label tracking-widest text-on-surface-variant/50 uppercase mb-3">
        {spec.title || 'Financial Trends'}
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ChartComp data={points} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#2a2c2b" />
          <XAxis dataKey="period" tick={{ fill: '#888', fontSize: 10 }} />
          <YAxis tick={{ fill: '#888', fontSize: 10 }} width={48} />
          <Tooltip contentStyle={{ background: '#1a1c1b', border: '1px solid #2a2c2b', borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {series.map((s, i) =>
            spec.chart_type === 'area' || spec.chart_type === 'stacked_area' ? (
              <Area
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                fill={CHART_COLORS[i % CHART_COLORS.length]}
                fillOpacity={0.18}
                strokeWidth={2}
                stackId={spec.chart_type === 'stacked_area' ? 'stack' : undefined}
              />
            ) : (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                strokeWidth={2.5}
                dot={false}
              />
            )
          )}
        </ChartComp>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Single turn bubble ──────────────────────────────────────────────────────

function TurnBubble({ turn, index }: { turn: ConversationTurn; index: number }) {
  const [agentExpanded, setAgentExpanded] = useState(false);

  const hasAgents = Object.keys(turn.agent_analyses || {}).length > 0;
  const firstTickerResult = turn.ticker_results?.[0];
  const hasSources = (firstTickerResult?.sources?.length ?? 0) > 0;
  const hasChart =
    !!firstTickerResult?.financial_data && !!firstTickerResult?.fundamentals_visualization?.charts?.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.04, 0.6) }}
      className="space-y-3"
    >
      {/* ── User bubble ─────────────────────────────────────────── */}
      <div className="flex items-end justify-end gap-2.5">
        <div className="flex flex-col items-end gap-1 max-w-[85%]">
          <div className="bg-primary/15 border border-primary/20 rounded-2xl rounded-br-md px-4 py-3 md:px-5 md:py-3.5 shadow-sm">
            <p className="text-sm md:text-base font-medium text-on-surface leading-relaxed">
              {turn.user_message}
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-on-surface-variant/50 font-medium pr-1">
            <Clock className="w-3 h-3" />
            <span>{fmtTimestamp(turn.created_at)}</span>
            {turn.duration_ms > 0 && (
              <>
                <span className="opacity-40">·</span>
                <span className="font-mono">{fmtDuration(turn.duration_ms)}</span>
              </>
            )}
          </div>
        </div>
        <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-surface-container-high border border-outline-variant/20 flex items-center justify-center shrink-0 shadow-sm">
          <User className="w-4 h-4 text-on-surface-variant" />
        </div>
      </div>

      {/* ── AI bubble ───────────────────────────────────────────── */}
      {turn.assistant_synthesis && (
        <div className="flex items-start gap-2.5">
          <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-primary/10 border border-primary/15 flex items-center justify-center shrink-0 shadow-sm mt-1">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div className="flex flex-col items-start gap-2 max-w-[92%] w-full">
            {/* Ticker badges */}
            {(turn.tickers ?? []).filter(Boolean).length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {turn.tickers.filter(Boolean).map((t) => (
                  <span
                    key={t}
                    className="inline-flex items-center gap-1 bg-surface-container-low border border-outline-variant/20 px-2 py-0.5 rounded-full text-[9px] md:text-[10px] font-black font-label tracking-wider text-primary uppercase"
                  >
                    <TrendingUp className="w-2.5 h-2.5" />
                    {t}
                  </span>
                ))}
              </div>
            )}

            {/* Synthesis card */}
            <div className="bg-surface-container border border-outline-variant/15 rounded-2xl rounded-bl-md px-4 py-3 md:px-5 md:py-4 shadow-sm w-full">
              <div className="prose prose-sm md:prose-base max-w-none prose-headings:font-headline prose-headings:font-bold prose-headings:text-on-surface prose-p:text-on-surface-variant prose-p:leading-relaxed prose-li:text-on-surface-variant prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-code:text-primary prose-code:bg-surface-container-highest prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs">
                <Markdown>{turn.assistant_synthesis}</Markdown>
              </div>

              {/* Inline chart */}
              {hasChart && (
                <TurnChart
                  financialData={firstTickerResult!.financial_data}
                  visualization={firstTickerResult!.fundamentals_visualization}
                />
              )}

              {/* Sources */}
              {hasSources && (
                <div className="mt-4 pt-4 border-t border-outline-variant/10">
                  <p className="text-[9px] font-black font-label tracking-widest text-on-surface-variant/50 uppercase mb-2 flex items-center gap-1.5">
                    <FileText className="w-3 h-3" /> Sources
                  </p>
                  <div className="space-y-1.5">
                    {firstTickerResult!.sources.map((src) => (
                      <a
                        key={src.source_id}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-2 group"
                      >
                        <span className="text-[10px] font-mono text-primary/60 shrink-0 mt-0.5">
                          [{src.source_id}]
                        </span>
                        <span className="text-[11px] text-on-surface-variant group-hover:text-primary transition-colors leading-snug">
                          {src.title}
                        </span>
                        <ExternalLink className="w-3 h-3 text-on-surface-variant/30 group-hover:text-primary transition-colors shrink-0 mt-0.5" />
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Agent analyses — expandable */}
            {hasAgents && (
              <div className="w-full">
                <button
                  onClick={() => setAgentExpanded((v) => !v)}
                  className="flex items-center gap-2 text-[10px] font-bold text-on-surface-variant/60 hover:text-on-surface-variant transition-colors uppercase tracking-wider pl-1"
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  Agent Reports ({Object.keys(turn.agent_analyses).length})
                  <ChevronDown
                    className={`w-3.5 h-3.5 transition-transform ${agentExpanded ? 'rotate-180' : ''}`}
                  />
                </button>

                <AnimatePresence>
                  {agentExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden mt-2 space-y-2"
                    >
                      {Object.entries(turn.agent_analyses).map(([agentKey, agentText]) => (
                        <div
                          key={agentKey}
                          className="bg-surface/80 border border-outline-variant/10 rounded-xl p-3"
                        >
                          <p className="text-[9px] font-black font-label tracking-widest text-primary uppercase mb-2">
                            {agentKey.replace(/_/g, ' ')}
                          </p>
                          <div className="prose prose-xs max-w-none prose-p:text-on-surface-variant/80 prose-p:text-[11px] prose-p:leading-relaxed prose-strong:text-on-surface prose-headings:text-on-surface prose-headings:text-xs prose-li:text-on-surface-variant/80 prose-li:text-[11px]">
                            <Markdown>{agentText}</Markdown>
                          </div>
                        </div>
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

// ─── Sticky Chat Input ───────────────────────────────────────────────────────

function FullViewChatInput({
  onAnalyze,
  isStreaming,
}: {
  onAnalyze: (q: string) => void;
  isStreaming: boolean;
}) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isStreaming) {
      onAnalyze(query.trim());
      setQuery('');
    }
  };

  return (
    <div className="shrink-0 border-t border-outline-variant/10 bg-surface-container-lowest/90 backdrop-blur-xl p-4 md:p-5">
      <form
        onSubmit={handleSubmit}
        className="max-w-4xl mx-auto bg-surface-container-lowest border border-outline-variant/20 rounded-[2.5rem] shadow-lg p-2 md:p-3 flex items-center gap-2 md:gap-4"
      >
        <button
          type="button"
          className="p-2 md:p-3.5 hover:bg-surface-container-low rounded-2xl text-on-surface-variant/60 transition-colors"
        >
          <Plus className="w-5 h-5" />
        </button>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isStreaming}
          placeholder="Continue the conversation…"
          className="flex-grow bg-transparent border-none focus:ring-0 text-sm font-semibold py-2 px-1 text-on-surface placeholder:text-on-surface-variant/40 outline-none"
        />
        <button
          type="submit"
          disabled={!query.trim() || isStreaming}
          className="bg-primary text-on-primary w-10 h-10 md:w-12 md:h-12 rounded-[1.25rem] flex items-center justify-center hover:shadow-[0_0_30px_rgba(0,200,5,0.35)] hover:scale-[1.02] transition-all active:scale-95 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          <ArrowUp className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
}

// ─── Main export ─────────────────────────────────────────────────────────────

interface ConversationFullViewProps {
  conversationId: string;
  turns: ConversationTurn[];
  onBack: () => void;
  onAnalyze: (query: string) => void;
  isStreaming?: boolean;
}

export default function ConversationFullView({
  conversationId,
  turns,
  onBack,
  onAnalyze,
  isStreaming = false,
}: ConversationFullViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Mark this conversation active so the next message continues it
  useEffect(() => {
    window.localStorage.setItem(STORAGE_CONVERSATION_ID, conversationId);
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationId, turns]);

  const shortId = `${conversationId.slice(0, 8)}…${conversationId.slice(-6)}`;
  const firstLabel =
    turns[0]?.user_message
      ? turns[0].user_message.length > 60
        ? `${turns[0].user_message.slice(0, 60)}…`
        : turns[0].user_message
      : shortId;

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 40 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="fixed inset-0 z-[80] bg-surface flex flex-col"
    >
      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="shrink-0 border-b border-outline-variant/10 bg-surface-container-lowest/80 backdrop-blur-xl px-4 md:px-8 py-3 md:py-4 flex items-center gap-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-sm font-bold text-on-surface-variant hover:text-on-surface transition-colors px-3 py-2 rounded-full hover:bg-surface-container-low"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <div className="min-w-0 flex-1">
          <p className="font-headline text-base md:text-lg font-bold text-on-surface truncate">{firstLabel}</p>
          <p className="text-[10px] text-on-surface-variant/50 font-medium">
            {turns.length} turns · {shortId}
          </p>
        </div>
        <div className="shrink-0 bg-primary/10 px-3 py-1 rounded-full">
          <span className="text-[10px] font-black text-primary uppercase tracking-wider font-label">
            Active
          </span>
        </div>
      </div>

      {/* ── Scrollable turn list ──────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 md:px-8 py-6 max-w-4xl w-full mx-auto space-y-8">
        {turns.length === 0 ? (
          <div className="flex items-center justify-center py-20 text-on-surface-variant/50 text-sm">
            No messages yet in this conversation.
          </div>
        ) : (
          turns.map((turn, i) => <TurnBubble key={turn.turn_id} turn={turn} index={i} />)
        )}
        <div ref={bottomRef} className="h-4" />
      </div>

      {/* ── Chat Input ────────────────────────────────────────────── */}
      <FullViewChatInput onAnalyze={onAnalyze} isStreaming={isStreaming} />
    </motion.div>
  );
}
