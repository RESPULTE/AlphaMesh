import { useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { User, Sparkles, Clock, TrendingUp, Eye } from 'lucide-react';
import Markdown from 'react-markdown';
import type { ConversationTurn } from '../types/api';

interface ConversationThreadProps {
  turns: ConversationTurn[];
  /** If true the container fills available height; otherwise uses a capped max-height */
  fullHeight?: boolean;
  /**
   * When set, shows only this many turns and renders a blurred overlay CTA below.
   * Set to 0 / undefined to show all turns.
   */
  maxPreviewTurns?: number;
  /** Called when the user clicks the "view full chatlog" overlay */
  onViewFull?: () => void;
}

const PREVIEW_LIMIT = 6;

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

interface TurnBubbleProps {
  turn: ConversationTurn;
  index: number;
  compact?: boolean;
}

function TurnBubble({ turn, index, compact = false }: TurnBubbleProps) {
  return (
    <motion.div
      key={turn.turn_id}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.04 }}
      className="space-y-3"
    >
      {/* ── User bubble ─────────────────────────────────────────────── */}
      <div className="flex items-end justify-end gap-2.5">
        <div className="flex flex-col items-end gap-1 max-w-[85%]">
          <div className="bg-primary/15 border border-primary/20 rounded-2xl rounded-br-md px-4 py-3 md:px-5 md:py-3.5 shadow-sm">
            <p className="text-sm md:text-base font-medium text-on-surface leading-relaxed">
              {turn.user_message}
            </p>
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
        <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-surface-container-high border border-outline-variant/20 flex items-center justify-center shrink-0 shadow-sm">
          <User className="w-4 h-4 text-on-surface-variant" />
        </div>
      </div>

      {/* ── AI bubble ───────────────────────────────────────────────── */}
      {turn.assistant_synthesis && (
        <div className="flex items-start gap-2.5">
          <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-primary/10 border border-primary/15 flex items-center justify-center shrink-0 shadow-sm">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div className="flex flex-col items-start gap-1.5 max-w-[92%]">
            {turn.tickers && turn.tickers.filter((t) => t?.trim()).length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {turn.tickers.filter((t) => t?.trim()).map((ticker) => (
                  <span
                    key={ticker}
                    className="inline-flex items-center gap-1 bg-surface-container-low border border-outline-variant/20 px-2 py-0.5 rounded-full text-[9px] md:text-[10px] font-black font-label tracking-wider text-primary uppercase"
                  >
                    <TrendingUp className="w-2.5 h-2.5" />
                    {ticker}
                  </span>
                ))}
              </div>
            )}
            <div className="bg-surface-container border border-outline-variant/15 rounded-2xl rounded-bl-md px-4 py-3 md:px-5 md:py-4 shadow-sm w-full">
              <div className={`prose max-w-none prose-headings:font-headline prose-headings:font-bold prose-headings:text-on-surface prose-p:text-on-surface-variant prose-p:leading-relaxed prose-li:text-on-surface-variant prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-code:text-primary prose-code:bg-surface-container-highest prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs ${compact ? 'prose-sm' : 'prose-sm md:prose-base'}`}>
                <Markdown>{turn.assistant_synthesis}</Markdown>
              </div>
            </div>
            {!compact && turn.agent_analyses && Object.keys(turn.agent_analyses).length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap pl-1">
                <span className="text-[9px] text-on-surface-variant/40 font-medium uppercase tracking-wider">Agents:</span>
                {Object.keys(turn.agent_analyses).map((agentKey) => (
                  <span
                    key={agentKey}
                    className="text-[9px] font-bold bg-surface-container-highest px-2 py-0.5 rounded-full text-on-surface-variant/70 font-label tracking-wide"
                  >
                    {agentKey.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}

export { TurnBubble };

export default function ConversationThread({
  turns,
  fullHeight = false,
  maxPreviewTurns,
  onViewFull,
}: ConversationThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const isPreviewMode = maxPreviewTurns !== undefined && maxPreviewTurns > 0;
  const visibleTurns = isPreviewMode ? turns.slice(0, maxPreviewTurns) : turns;
  const hiddenCount = isPreviewMode ? Math.max(0, turns.length - maxPreviewTurns) : 0;

  useEffect(() => {
    if (!isPreviewMode) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns, isPreviewMode]);

  if (!turns.length) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-on-surface-variant/60">
        No messages in this conversation yet.
      </div>
    );
  }

  return (
    <div className={`relative ${fullHeight ? 'flex-1' : ''}`}>
      <div
        className={`overflow-y-auto custom-scrollbar space-y-8 p-4 md:p-6 ${
          fullHeight ? 'flex-1' : 'max-h-[70vh]'
        }`}
      >
        {visibleTurns.map((turn, index) => (
          <TurnBubble key={turn.turn_id} turn={turn} index={index} />
        ))}
        {!isPreviewMode && <div ref={bottomRef} className="h-1" />}
      </div>

      {/* ── Blurred overlay when in preview mode with hidden turns ── */}
      {isPreviewMode && hiddenCount > 0 && (
        <div
          className="absolute bottom-0 left-0 right-0 h-40 flex flex-col items-center justify-end pb-5 cursor-pointer"
          style={{
            background:
              'linear-gradient(to bottom, transparent 0%, rgba(var(--surface-container-lowest-rgb, 18,20,20), 0.85) 60%, rgba(var(--surface-container-lowest-rgb, 18,20,20), 0.97) 100%)',
            backdropFilter: 'blur(4px)',
            WebkitBackdropFilter: 'blur(4px)',
          }}
          onClick={onViewFull}
        >
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center gap-2"
          >
            <div className="flex items-center gap-2 bg-primary/10 border border-primary/20 text-primary px-4 py-2 rounded-full text-xs font-bold font-label tracking-wider uppercase hover:bg-primary/20 transition-colors">
              <Eye className="w-3.5 h-3.5" />
              View full chatlog · {hiddenCount} more turn{hiddenCount > 1 ? 's' : ''}
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
