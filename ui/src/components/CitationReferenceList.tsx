import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState, type ForwardedRef } from 'react';
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import clsx from 'clsx';
import type { SourceItem } from '../types/api';
import { groupSourcesByArticle } from '../utils/sourceGrouping';

const NO_CHUNK_TEXT_FALLBACK = '(No chunk text available)';

export interface CitationReferenceListHandle {
  focusCitation: (citationId: number) => void;
  scrollToTop: () => void;
}

interface CitationReferenceListProps {
  sources: SourceItem[];
  className?: string;
}

function _CitationReferenceList(
  { sources, className }: CitationReferenceListProps,
  ref: ForwardedRef<CitationReferenceListHandle>
) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const titleClickTimerRef = useRef<number | null>(null);
  const groupedReferences = useMemo(() => groupSourcesByArticle(sources || []), [sources]);
  const [expandedReferenceKey, setExpandedReferenceKey] = useState<string | null>(null);
  const [activeCitationId, setActiveCitationId] = useState<number | null>(null);

  useEffect(() => {
    return () => {
      if (titleClickTimerRef.current !== null) {
        window.clearTimeout(titleClickTimerRef.current);
      }
    };
  }, []);

  const scrollToTop = () => {
    sectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const focusCitation = (citationId: number) => {
    scrollToTop();
    const target = groupedReferences.find((item) => item.citationIds.includes(citationId));
    if (!target) {
      return;
    }
    setExpandedReferenceKey(target.key);
    setActiveCitationId(citationId);
  };

  useImperativeHandle(ref, () => ({ scrollToTop, focusCitation }), [groupedReferences]);

  if (!groupedReferences.length) {
    return null;
  }

  const resolveActiveCitationForReference = (referenceKey: string, citationIds: number[]) => {
    if (!citationIds.length) return null;
    if (expandedReferenceKey === referenceKey && activeCitationId !== null && citationIds.includes(activeCitationId)) {
      return activeCitationId;
    }
    return citationIds[0];
  };

  const toggleReference = (referenceKey: string) => {
    setExpandedReferenceKey((current) => (current === referenceKey ? null : referenceKey));
    const target = groupedReferences.find((item) => item.key === referenceKey);
    if (target?.citationIds?.length) {
      setActiveCitationId(target.citationIds[0]);
    } else {
      setActiveCitationId(null);
    }
  };

  const handleTitleClick = (referenceKey: string) => {
    if (titleClickTimerRef.current !== null) {
      window.clearTimeout(titleClickTimerRef.current);
      titleClickTimerRef.current = null;
    }
    titleClickTimerRef.current = window.setTimeout(() => {
      toggleReference(referenceKey);
      titleClickTimerRef.current = null;
    }, 220);
  };

  const handleTitleDoubleClick = (url: string) => {
    if (titleClickTimerRef.current !== null) {
      window.clearTimeout(titleClickTimerRef.current);
      titleClickTimerRef.current = null;
    }
    if (!url) return;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div ref={sectionRef} className={clsx('space-y-3', className)}>
      {groupedReferences.map((reference) => {
        const isExpanded = expandedReferenceKey === reference.key;
        const effectiveCitationId = resolveActiveCitationForReference(
          reference.key,
          reference.citationIds
        );
        const evidenceRows = reference.citationIds
          .map((citationId) => ({
            citationId,
            text:
              reference.chunkByCitationId[citationId] ||
              (citationId === reference.citationIds[0] ? reference.primaryChunkText : '') ||
              NO_CHUNK_TEXT_FALLBACK,
          }))
          .filter((row) => row.text);
        const isActiveTarget =
          activeCitationId !== null && reference.citationIds.includes(activeCitationId);

        return (
          <div
            key={reference.key}
            className={clsx(
              'rounded-2xl border bg-surface-container-lowest transition-all',
              isActiveTarget
                ? 'border-primary/50 shadow-[0_0_18px_rgba(0,200,5,0.2)]'
                : 'border-outline-variant/20'
            )}
          >
            <div className="p-4">
              <div className="flex items-start gap-3">
                <button
                  type="button"
                  className="min-w-7 h-7 rounded-full bg-surface-container-high text-on-surface-variant text-xs font-bold font-mono px-1 hover:bg-primary/10 hover:text-primary transition-colors"
                  onClick={() => toggleReference(reference.key)}
                  aria-label={`Toggle reference ${reference.title}`}
                >
                  {reference.citationIds[0] ?? '?'}
                </button>
                <div className="flex-1 min-w-0">
                  <button
                    type="button"
                    className="w-full text-left text-sm font-medium text-on-surface hover:text-primary transition-colors leading-snug"
                    onClick={() => handleTitleClick(reference.key)}
                    onDoubleClick={() => handleTitleDoubleClick(reference.url)}
                    title="Single-click to expand, double-click to open article"
                  >
                    {reference.title}
                  </button>
                  <div className="mt-1.5 flex items-center gap-2 text-[11px] text-on-surface-variant/60 font-mono uppercase tracking-wider">
                    <span>{reference.domain}</span>
                    {reference.url && (
                      <span className="inline-flex items-center gap-1 text-primary/80 normal-case tracking-normal">
                        <ExternalLink className="w-3 h-3" />
                        dbl-click title to open
                      </span>
                    )}
                  </div>
                  {reference.citationIds.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {reference.citationIds.map((citationId) => (
                        <button
                          key={`${reference.key}-cid-${citationId}`}
                          type="button"
                          onClick={() => {
                            setExpandedReferenceKey(reference.key);
                            setActiveCitationId(citationId);
                          }}
                          className={clsx(
                            'text-[11px] font-mono px-2 py-0.5 rounded-full border transition-colors',
                            citationId === effectiveCitationId
                              ? 'border-primary/70 bg-primary/18 text-primary shadow-[0_0_16px_rgba(0,200,5,0.2)]'
                              : 'border-primary/35 bg-primary/8 text-primary/90 hover:border-primary/55 hover:bg-primary/14'
                          )}
                        >
                          [{citationId}]
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => toggleReference(reference.key)}
                  className="text-on-surface-variant/60 hover:text-primary transition-colors"
                  aria-label={isExpanded ? 'Collapse reference' : 'Expand reference'}
                >
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {isExpanded && (
              <div
                className={clsx(
                  'border-t px-4 py-3',
                  isActiveTarget
                    ? 'border-primary/30 bg-primary/5'
                    : 'border-outline-variant/10 bg-surface-container-low/40'
                )}
              >
                <div className="text-[10px] font-black font-label tracking-wider uppercase text-on-surface-variant/60 mb-1.5">
                  Cited chunk text
                </div>
                <div className="space-y-3">
                  {(evidenceRows.length ? evidenceRows : [{ citationId: effectiveCitationId ?? -1, text: NO_CHUNK_TEXT_FALLBACK }]).map(
                    (row) => (
                      <div
                        key={`${reference.key}-evidence-${row.citationId}`}
                        className={clsx(
                          'rounded-lg border px-3 py-2',
                          row.citationId === effectiveCitationId
                            ? 'border-primary/40 bg-primary/8'
                            : 'border-outline-variant/15 bg-surface-container-low/20'
                        )}
                      >
                        {row.citationId > 0 && (
                          <div className="text-[10px] font-mono text-primary mb-1">
                            [{row.citationId}]
                          </div>
                        )}
                        <p className="text-xs text-on-surface-variant leading-relaxed whitespace-pre-wrap">
                          {row.text}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

const CitationReferenceList = forwardRef<CitationReferenceListHandle, CitationReferenceListProps>(
  _CitationReferenceList
);

export default CitationReferenceList;
