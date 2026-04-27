import { useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { User, Sparkles, Clock, TrendingUp } from 'lucide-react';
import Markdown from 'react-markdown';
import type { ConversationTurn } from '../types/api';

interface ConversationThreadProps {
  turns: ConversationTurn[];
  /** If true the container fills available height; otherwise uses a capped max-height */
  fullHeight?: boolean;
}

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

export default function ConversationThread({
  turns,
  fullHeight = false,
}: ConversationThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to the bottom of the thread whenever turns change
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns]);

  if (!turns.length) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-on-surface-variant/60">
        No messages in this conversation yet.
      </div>
    );
  }

  return (
    <div
      className={`overflow-y-auto custom-scrollbar space-y-8 p-4 md:p-6 ${
        fullHeight ? 'flex-1' : 'max-h-[70vh]'
      }`}
    >
      {turns.map((turn, index) => (
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
              {/* Timestamp + duration */}
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
            {/* User avatar */}
            <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-surface-container-high border border-outline-variant/20 flex items-center justify-center shrink-0 shadow-sm">
              <User className="w-4 h-4 text-on-surface-variant" />
            </div>
          </div>

          {/* ── AI bubble ───────────────────────────────────────────────── */}
          {turn.assistant_synthesis && (
            <div className="flex items-start gap-2.5">
              {/* AI avatar */}
              <div className="w-8 h-8 md:w-9 md:h-9 rounded-full bg-primary/10 border border-primary/15 flex items-center justify-center shrink-0 shadow-sm">
                <Sparkles className="w-4 h-4 text-primary" />
              </div>
              <div className="flex flex-col items-start gap-1.5 max-w-[92%]">
                {/* Tickers badge row */}
                {turn.tickers && turn.tickers.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {turn.tickers
                      .filter((t) => t && t.trim())
                      .map((ticker) => (
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

                {/* Synthesis text */}
                <div className="bg-surface-container border border-outline-variant/15 rounded-2xl rounded-bl-md px-4 py-3 md:px-5 md:py-4 shadow-sm w-full">
                  <div className="prose prose-sm md:prose-base max-w-none prose-headings:font-headline prose-headings:font-bold prose-headings:text-on-surface prose-p:text-on-surface-variant prose-p:leading-relaxed prose-li:text-on-surface-variant prose-a:text-primary hover:prose-a:text-primary/80 prose-strong:text-on-surface prose-code:text-primary prose-code:bg-surface-container-highest prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs">
                    <Markdown>{turn.assistant_synthesis}</Markdown>
                  </div>
                </div>

                {/* Agent chips */}
                {turn.agent_analyses && Object.keys(turn.agent_analyses).length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap pl-1">
                    <span className="text-[9px] text-on-surface-variant/40 font-medium uppercase tracking-wider">
                      Agents:
                    </span>
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
      ))}

      {/* Scroll anchor */}
      <div ref={bottomRef} className="h-1" />
    </div>
  );
}
