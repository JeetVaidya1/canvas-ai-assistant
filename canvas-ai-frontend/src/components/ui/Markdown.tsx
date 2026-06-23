import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark.css'

interface MarkdownProps {
  content: string
  /** Extra classes for the wrapper (e.g. text sizing per surface). */
  className?: string
}

/**
 * Renders AI/markdown text with GitHub-flavored markdown, KaTeX math, and
 * syntax-highlighted code. Styling lives in the `.markdown-body` scope in
 * index.css so it stays consistent across chat, notes, quiz, practice, etc.
 */
export function Markdown({ content, className = '' }: MarkdownProps) {
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
