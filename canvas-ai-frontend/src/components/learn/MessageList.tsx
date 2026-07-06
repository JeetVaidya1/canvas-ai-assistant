import { useState } from 'react'
import type { RefObject } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'
import { CitedMarkdown } from '@/components/learn/CitedMarkdown'
import { SourcesDisclosure } from '@/components/learn/SourcesDisclosure'
import type { Message } from '@/lib/api'

// After an answer, these keep the study loop going (Claude-style follow-ups).
const FOLLOWUPS: ReadonlyArray<{ label: string; prompt: string }> = [
  { label: 'Explain simpler', prompt: 'Explain that again in simpler terms, with an analogy.' },
  { label: 'Give an example', prompt: 'Give me a concrete worked example of that.' },
  { label: 'Quiz me on this', prompt: 'Ask me 3 quick questions to test my understanding of what you just explained.' },
]

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Copy-to-clipboard action with a transient check. */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        void navigator.clipboard.writeText(text)
        setCopied(true)
        window.setTimeout(() => setCopied(false), 1400)
      }}
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] text-ink-faint transition-colors hover:bg-paper-deep hover:text-ink"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

interface MessageListProps {
  messages: ReadonlyArray<Message>
  isTyping: boolean
  showFollowups: boolean
  onFollowup: (prompt: string) => void
  bottomRef: RefObject<HTMLDivElement | null>
}

/** The Learn chat transcript: prose answers, inline citations, sources, follow-ups. */
export function MessageList({ messages, isTyping, showFollowups, onFollowup, bottomRef }: MessageListProps) {
  return (
    <div className="mx-auto w-full max-w-3xl px-5 py-8">
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className={cn('group', msg.role === 'user' ? 'mb-6 flex justify-end' : 'mb-8')}
          >
            {msg.role === 'user' ? (
              <div className="max-w-[80%] rounded-2xl rounded-br-md border border-line bg-paper-deep px-4 py-2.5 text-[15px] leading-relaxed text-ink">
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            ) : (
              <div className="min-w-0">
                <CitedMarkdown content={msg.content} className="text-[15px] leading-relaxed" />
                {msg.sources && msg.sources.length > 0 && <SourcesDisclosure sources={msg.sources} />}
                <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <CopyButton text={msg.content} />
                  <span className="text-[11px] text-ink-faint">{formatTime(msg.timestamp)}</span>
                </div>
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>

      {isTyping && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-8 flex items-center gap-1.5">
          <span className="text-sm text-ink-soft">Searching your materials</span>
          <span className="flex gap-1">
            {[0, 0.15, 0.3].map((delay) => (
              <span
                key={delay}
                className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent/80"
                style={{ animationDelay: `${delay}s` }}
              />
            ))}
          </span>
        </motion.div>
      )}

      {/* Follow-up suggestions keep the study loop going */}
      {showFollowups && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="flex flex-wrap gap-2"
        >
          {FOLLOWUPS.map(({ label, prompt }) => (
            <button
              key={label}
              onClick={() => onFollowup(prompt)}
              className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1.5 text-[13px] text-ink-soft transition-all hover:border-accent-line hover:bg-accent-wash hover:text-accent-deep"
            >
              {label}
            </button>
          ))}
        </motion.div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
