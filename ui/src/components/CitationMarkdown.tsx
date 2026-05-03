import Markdown from 'react-markdown';

type MarkdownNode = {
  type?: string;
  value?: string;
  children?: MarkdownNode[];
  url?: string;
  title?: string | null;
};

type ReactLikeNode = unknown;

interface CitationMarkdownProps {
  text: string;
  className?: string;
  onCitationClick?: (citationId: number) => void;
}

const CITATION_PATTERN = /\[(\d+)\]/g;
const HAS_CITATION_PATTERN = /\[(\d+)\]/;
const ONLY_CITATION_PATTERN = /^\[(\d+)\]$/;

function isSafeExternalHref(href: string): boolean {
  const value = (href || '').trim().toLowerCase();
  return (
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('mailto:') ||
    value.startsWith('tel:')
  );
}

function extractNodeText(node: ReactLikeNode): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(extractNodeText).join('');
  }
  if (node && typeof node === 'object') {
    const maybeProps = (node as { props?: { children?: ReactLikeNode } }).props;
    if (maybeProps && 'children' in maybeProps) {
      return extractNodeText(maybeProps.children);
    }
  }
  return '';
}

function splitTextIntoCitationNodes(text: string): MarkdownNode[] {
  const nodes: MarkdownNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null = null;
  CITATION_PATTERN.lastIndex = 0;

  while ((match = CITATION_PATTERN.exec(text)) !== null) {
    const start = match.index;
    const end = CITATION_PATTERN.lastIndex;
    const citationId = Number(match[1]);
    if (start > lastIndex) {
      nodes.push({ type: 'text', value: text.slice(lastIndex, start) });
    }
    nodes.push({
      type: 'link',
      url: `citation:${citationId}`,
      title: null,
      children: [{ type: 'text', value: `[${citationId}]` }],
    });
    lastIndex = end;
  }

  if (lastIndex < text.length) {
    nodes.push({ type: 'text', value: text.slice(lastIndex) });
  }
  return nodes.length ? nodes : [{ type: 'text', value: text }];
}

function transformCitationLinks(node: MarkdownNode) {
  if (!Array.isArray(node.children) || node.children.length === 0) {
    return;
  }

  const nextChildren: MarkdownNode[] = [];
  for (const child of node.children) {
    if (child?.type === 'text' && typeof child.value === 'string' && HAS_CITATION_PATTERN.test(child.value)) {
      nextChildren.push(...splitTextIntoCitationNodes(child.value));
      continue;
    }
    transformCitationLinks(child);
    nextChildren.push(child);
  }
  node.children = nextChildren;
}

function remarkCitationLinks() {
  return (tree: MarkdownNode) => {
    transformCitationLinks(tree);
  };
}

export default function CitationMarkdown({
  text,
  className,
  onCitationClick,
}: CitationMarkdownProps) {
  return (
    <div className={className}>
      <Markdown
        remarkPlugins={[remarkCitationLinks]}
        components={{
          a: ({ href, children }) => {
            const rawHref = String(href || '').trim();
            const childText = extractNodeText(children).trim();
            const citationFromHref = rawHref.startsWith('citation:')
              ? Number(rawHref.replace('citation:', ''))
              : NaN;
            const citationFromTextMatch = childText.match(ONLY_CITATION_PATTERN);
            const citationFromText = citationFromTextMatch
              ? Number(citationFromTextMatch[1])
              : NaN;
            const citationId = Number.isFinite(citationFromHref)
              ? citationFromHref
              : citationFromText;

            if (Number.isFinite(citationId)) {
              return (
                <button
                  type="button"
                  className="citation-chip"
                  onClick={() => {
                    if (Number.isFinite(citationId)) {
                      onCitationClick?.(citationId);
                    }
                  }}
                  aria-label={`Jump to citation ${citationId}`}
                >
                  {children}
                </button>
              );
            }
            if (!isSafeExternalHref(rawHref)) {
              return <span>{children}</span>;
            }
            return (
              <a href={rawHref} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {text || ''}
      </Markdown>
    </div>
  );
}
