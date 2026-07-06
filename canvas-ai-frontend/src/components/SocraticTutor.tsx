import { useState, useRef, useEffect } from 'react'
import type { FormEvent } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Send, User, Compass, MessageCircleQuestion, Check } from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { ErrorState } from '@/components/ui/States'
import { socraticTurn, type TutorTurn } from '@/lib/api'

interface SocraticTutorProps {
  courseId: string
}

const STARTERS = [
  'Walk me through this problem',
  'Help me understand this proof',
  'I keep getting this wrong',
] as const

const PROMISES = [
  { icon: MessageCircleQuestion, text: 'I ask questions instead of handing you answers' },
  { icon: Compass, text: 'Every nudge is grounded in your course materials' },
  { icon: Check, text: "We stop when it clicks — not before" },
] as const

export default function SocraticTutor({ courseId }: SocraticTutorProps) {
  const [turns, setTurns] = useState<TutorTurn[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  /** The user message that failed to get a reply — drives the retry UI. */
  const [failedMessage, setFailedMessage] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, thinking, failedMessage])

  const runTurn = async (msg: string, history: TutorTurn[]) => {
    setThinking(true)
    setFailedMessage(null)
    try {
      const { reply } = await socraticTurn(courseId, msg, history)
      setTurns((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch {
      setFailedMessage(msg)
    } finally {
      setThinking(false)
    }
  }

  const send = (e?: FormEvent) => {
    if (e) e.preventDefault()
    const msg = input.trim()
    if (!msg || !courseId || thinking) return
    setInput('')
    const history = turns
    setTurns((prev) => [...prev, { role: 'user', content: msg }])
    void runTurn(msg, history)
  }

  const retry = () => {
    if (!failedMessage || thinking) return
    // The failed user turn is already in the transcript — resend against the
    // history that preceded it.
    void runTurn(failedMessage, turns.slice(0, -1))
  }

  const empty = turns.length === 0 && !thinking

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto px-5">
        {empty ? (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: 'easeOut' }}
            className="max-w-2xl mx-auto text-center py-12"
          >
            <div className="relative w-16 h-16 mx-auto mb-6">
              <BrandMark className="absolute inset-0 h-full w-full" />
            </div>
            <h3 className="font-display text-2xl font-semibold text-ink mb-3">
              I'll guide you — not hand you the answer
            </h3>
            <p className="text-sm text-ink-soft leading-relaxed mb-7 max-w-md mx-auto">
              Tell me what you're working on or stuck on. I'll ask the right questions until the idea
              becomes yours.
            </p>

            <div className="space-y-2.5 text-left mb-8 max-w-xl mx-auto">
              {PROMISES.map(({ icon: Icon, text }, i) => (
                <motion.div
                  key={text}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.15 + i * 0.08, duration: 0.3 }}
                  className="flex items-center gap-3 rounded-xl card-surface px-4 py-2.5"
                >
                  <span className="w-7 h-7 rounded-lg bg-accent-wash border border-accent-line flex items-center justify-center flex-shrink-0">
                    <Icon className="w-3.5 h-3.5 text-accent" />
                  </span>
                  <span className="text-sm text-ink-soft">{text}</span>
                </motion.div>
              ))}
            </div>

            <div className="flex flex-wrap justify-center gap-2">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-[13px] text-ink-soft transition-all hover:border-accent-line hover:bg-accent-wash hover:text-accent-deep"
                >
                  {s}
                </button>
              ))}
            </div>
          </motion.div>
        ) : (
          <div className="space-y-5 max-w-3xl mx-auto py-6">
            <AnimatePresence initial={false}>
              {turns.map((t, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className={`flex items-start gap-3 ${t.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border ${
                      t.role === 'user'
                        ? 'bg-paper-deep border-line'
                        : 'bg-accent-wash border-accent-line'
                    }`}
                  >
                    {t.role === 'user' ? (
                      <User className="w-4 h-4 text-ink-soft" />
                    ) : (
                      <Compass className="w-4 h-4 text-accent" />
                    )}
                  </div>
                  <div className={`max-w-[85%] ${t.role === 'user' ? 'text-right' : ''}`}>
                    {t.role === 'assistant' && (
                      <p className="text-[11px] font-medium tracking-wider text-ink-faint mb-1 ml-1">
                        Tutor
                      </p>
                    )}
                    <div
                      className={`inline-block px-4 py-3 rounded-2xl ${
                        t.role === 'user'
                          ? 'bg-paper-deep border border-line text-ink rounded-tr-sm'
                          : 'card-surface rounded-tl-sm'
                      }`}
                    >
                      {t.role === 'user' ? (
                        <p className="whitespace-pre-wrap leading-relaxed text-sm">{t.content}</p>
                      ) : (
                        <Markdown content={t.content} className="text-sm" />
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {thinking && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start gap-3"
              >
                <div className="w-8 h-8 rounded-full bg-accent-wash border border-accent-line flex items-center justify-center">
                  <Compass className="w-4 h-4 text-accent" />
                </div>
                <div className="card-surface rounded-2xl rounded-tl-sm px-4 py-3.5">
                  <div className="flex space-x-1.5">
                    {[0, 0.15, 0.3].map((d) => (
                      <motion.span
                        key={d}
                        className="w-2 h-2 bg-accent/70 rounded-full"
                        animate={{ y: [0, -4, 0], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 0.9, repeat: Infinity, delay: d }}
                      />
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {failedMessage && (
              <ErrorState
                compact
                title="The tutor couldn't respond."
                onRetry={retry}
                retrying={thinking}
              />
            )}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <form onSubmit={send} className="flex-shrink-0 w-full px-5 pb-4 pt-2">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 rounded-2xl card-surface p-2 focus-within:border-accent transition-colors">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="What are you working on?"
              disabled={thinking}
              rows={1}
              className="flex-1 resize-none bg-transparent px-3 py-2 text-sm text-ink placeholder-ink-faint outline-none max-h-32"
            />
            <Button
              type="submit"
              loading={thinking}
              disabled={thinking || !input.trim()}
              aria-label="Send message"
              className="px-3 self-end"
            >
              {!thinking && <Send className="w-4 h-4" />}
            </Button>
          </div>
          <p className="text-[11px] text-ink-faint text-center mt-2">
            Enter to send · Shift+Enter for a new line
          </p>
        </div>
      </form>
    </div>
  )
}
