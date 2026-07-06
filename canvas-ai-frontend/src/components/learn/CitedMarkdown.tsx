import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark.css'
import { CitationChip } from '@/components/learn/CitationChip'
import { fileLabel } from '@/components/learn/citation-utils'

/**
 * Markdown renderer for AI answers that prettifies raw RAG citation markers.
 *
 * The backend labels context chunks like `[1:lecture3.pdf:12]` (or
 * `[V2:slides.pptx:4]` for visual chunks, `[1: notes.pdf p12]` from the legacy
 * engine) and, despite prompt instructions, models sometimes leak those tags
 * into the streamed answer. This component rewrites each marker into a
 * markdown link with a `#cite=` fragment href, then renders those links as
 * inline citation chips (Badge + Tooltip). Everything else matches the
 * `Markdown` primitive (GFM, KaTeX, highlighted code, `.markdown-body` scope).
 */

const CITE_HREF_PREFIX = '#cite='

/** `[1:file:page]`, `[V2:file]`, `[3: file p12]` — index prefix + payload. */
const MARKER_RE = /\[\s*V?\d+\s*:\s*([^\][]+?)\s*\]/g

/** Split out fenced blocks and inline code so markers inside code stay raw. */
const CODE_SPLIT_RE = /(```[\s\S]*?```|`[^`\n]*`)/g

interface ParsedCite {
  file: string
  page?: number
}

function parseCitePayload(inner: string): ParsedCite | null {
  let file = inner.trim()
  let page: number | undefined

  const colonIdx = file.lastIndexOf(':')
  if (colonIdx > 0) {
    const tail = file.slice(colonIdx + 1).trim()
    const pageMatch = tail.match(/^(?:p\.?\s*|page\s*|slide\s*)?(\d+)$/i)
    if (pageMatch) {
      page = Number(pageMatch[1])
      file = file.slice(0, colonIdx).trim()
    }
  }
  if (page === undefined) {
    const spaced = file.match(/^(.+?)\s+(?:p\.?|page|slide)\s*(\d+)$/i)
    if (spaced) {
      file = spaced[1].trim()
      page = Number(spaced[2])
    }
  }

  // Guard against non-citation brackets like timestamps `[12:30]`.
  if (!/[a-zA-Z]/.test(file)) return null
  return { file, page }
}

function toCiteHref({ file, page }: ParsedCite): string {
  return `${CITE_HREF_PREFIX}${encodeURIComponent(file)}${page !== undefined ? `&page=${page}` : ''}`
}

function fromCiteHref(href: string): ParsedCite {
  const raw = href.slice(CITE_HREF_PREFIX.length)
  const [encodedFile, pagePart] = raw.split('&page=')
  return {
    file: decodeURIComponent(encodedFile),
    page: pagePart !== undefined ? Number(pagePart) : undefined,
  }
}

/** Rewrite leaked `[i:file:page]` markers into `#cite=` markdown links. */
function rewriteCitationMarkers(content: string): string {
  return content
    .split(CODE_SPLIT_RE)
    .map((segment, i) => {
      const isCode = i % 2 === 1
      if (isCode) return segment
      return segment.replace(MARKER_RE, (whole, inner: string) => {
        const cite = parseCitePayload(inner)
        if (!cite) return whole
        return `[${fileLabel(cite.file)}](${toCiteHref(cite)})`
      })
    })
    .join('')
}

interface CitedMarkdownProps {
  content: string
  /** Extra classes for the wrapper (e.g. text sizing per surface). */
  className?: string
}

export function CitedMarkdown({ content, className = '' }: CitedMarkdownProps) {
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          a: ({ href, children, title }) => {
            if (href?.startsWith(CITE_HREF_PREFIX)) {
              const { file, page } = fromCiteHref(href)
              return <CitationChip file={file} page={page} />
            }
            return (
              <a href={href} title={title} target="_blank" rel="noreferrer">
                {children}
              </a>
            )
          },
        }}
      >
        {rewriteCitationMarkers(content)}
      </ReactMarkdown>
    </div>
  )
}
