import type { SourceItem } from '../types/api';

export interface GroupedArticleSource {
  key: string;
  title: string;
  url: string;
  domain: string;
  citationIds: number[];
}

const _WHITESPACE_RE = /\s+/g;

export function canonicalizeSourceUrl(url: string): string {
  const raw = (url || '').trim();
  if (!raw) return '';
  try {
    const parsed = new URL(raw);
    const protocol = parsed.protocol.toLowerCase();
    const host = parsed.host.toLowerCase();
    let pathname = parsed.pathname || '';
    if (pathname !== '/' && pathname.endsWith('/')) {
      pathname = pathname.slice(0, -1);
    }
    return `${protocol}//${host}${pathname}`.trim();
  } catch {
    return raw.split('?', 1)[0].split('#', 1)[0].trim();
  }
}

export function normalizeSourceTitle(title: string): string {
  const raw = (title || '').trim().toLowerCase();
  if (!raw) return '';
  return raw.replace(_WHITESPACE_RE, ' ');
}

export function extractSourceDomain(url: string): string {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, '');
  } catch {
    return 'Source';
  }
}

function _sourceGroupKey(source: Pick<SourceItem, 'title' | 'url'>): string {
  const canonicalUrl = canonicalizeSourceUrl(source.url);
  if (canonicalUrl) return `url:${canonicalUrl}`;
  const normalizedTitle = normalizeSourceTitle(source.title) || 'unknown title';
  return `title:${normalizedTitle}`;
}

export function groupSourcesByArticle(
  sources: Array<Pick<SourceItem, 'source_id' | 'title' | 'url'>>
): GroupedArticleSource[] {
  const grouped = new Map<string, GroupedArticleSource>();

  for (const source of sources) {
    const key = _sourceGroupKey(source);
    const title = (source.title || '').trim() || 'Unknown Title';
    const url = (source.url || '').trim();
    const citationId = Number(source.source_id);

    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        key,
        title,
        url,
        domain: extractSourceDomain(url),
        citationIds: Number.isFinite(citationId) ? [citationId] : [],
      });
      continue;
    }

    if (Number.isFinite(citationId) && !existing.citationIds.includes(citationId)) {
      existing.citationIds.push(citationId);
    }
    if (!existing.url && url) {
      existing.url = url;
      existing.domain = extractSourceDomain(url);
    }
    if (existing.title === 'Unknown Title' && title !== 'Unknown Title') {
      existing.title = title;
    }
  }

  for (const article of grouped.values()) {
    article.citationIds.sort((a, b) => a - b);
  }

  return Array.from(grouped.values());
}
