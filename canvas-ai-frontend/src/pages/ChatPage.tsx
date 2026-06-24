import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'motion/react'
import {
  ArrowUp,
  Plus,
  Trash2,
  History,
  FileText,
  ChevronDown,
  Copy,
  Check,
  X,
  GraduationCap,
  BookOpen,
  Lightbulb,
  Wand2,
} from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { BrandMark } from '@/components/ui/BrandMark'
import { cn } from '@/lib/utils'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { trackVisit } from '@/hooks/useRecentActivity'
import {
  askQuestionStream,
  getChatSessions,
  getSessionMessages,
  deleteSession,
  type ChatSession,
  type Message,
  type Source,
} from '@/lib/api'

const STARTERS: ReadonlyArray<{ label: string; prompt: string }> = [
  { label: 'Summarize the course', prompt: 'Summarize the key concepts from my course materials.' },
  { label: 'Explain the hardest idea', prompt: 'Explain the hardest concept in simple terms, with an example.' },
  { label: 'Quiz me', prompt: 'Quiz me on the most important topics in this course.' },
  { label: 'Connect the themes', prompt: 'What are the main themes and how do they connect?' },
]

// After an answer, these keep the study loop going (Claude-style follow-ups).
const FOLLOWUPS: ReadonlyArray<{ icon: typeof BookOpen; label: string; prompt: string }> = [
  { icon: Lightbulb, label: 'Explain simpler', prompt: 'Explain that again in simpler terms, with an analogy.' },
  { icon: BookOpen, label: 'Give an example', prompt: 'Give me a concrete worked example of that.' },
  { icon: Wand2, label: 'Quiz me on this', prompt: 'Ask me 3 quick questions to test my understanding of what you just explained.' },
]

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function fileLabel(file: string): string {
  const parts = file.split(/[/\\]/)
  return parts[parts.length - 1] || file
}

/** Collapsible source citations — tidy "N sources" pill that expands to cited pages. */
function Sources({ sources }: { sources: ReadonlyArray<Source> }) {
  const [open, setOpen] = useState(false)
  if (sources.length === 0) return null
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-full border border-cyan-400/20 bg-cyan-500/[0.08] px-2.5 py-1 text-[11px] font-medium text-cyan-200 transition-colors hover:bg-cyan-500/15"
      >
        <FileText className="h-3 w-3" />
        {sources.length} source{sources.length !== 1 ? 's' : ''} from your materials
        <ChevronDown className={cn('h-3 w-3 transition-transform', open && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex flex-wrap gap-1.5">
              {sources.map((s, i) => (
                <span
                  key={`${s.file}-${s.page ?? 'x'}-${i}`}
                  title={s.file}
                  className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-zinc-300"
                >
                  <FileText className="h-3 w-3 text-cyan-400/80 flex-shrink-0" />
                  <span className="truncate max-w-[200px]">{fileLabel(s.file)}</span>
                  {s.page != null && (
                    <span className="rounded bg-cyan-500/15 px-1 text-[10px] font-semibold text-cyan-200">
                      p.{s.page}
                    </span>
                  )}
                </span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
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
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-zinc-200"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

export default function ChatPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const { data: files } = useCourseFiles(courseId)

  const course = courses?.find((c) => c.course_id === courseId)

  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const loadSessions = useCallback(async () => {
    try {
      const data = await getChatSessions(userId)
      setSessions(data || [])
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [userId])

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'chat')
  }, [courseId])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!activeSession) return
      try {
        const data = await getSessionMessages(activeSession.id)
        if (!cancelled) setMessages(data || [])
      } catch (e) {
        console.error('Failed to load messages:', e)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [activeSession])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // Auto-grow the composer textarea.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [question])

  const handleAsk = async (e?: React.FormEvent, presetText?: string) => {
    if (e) e.preventDefault()
    const trimmed = (presetText ?? question).trim()
    if (!trimmed || !courseId || isTyping) return

    setQuestion('')
    setIsTyping(true)

    const userMessage: Message = {
      id: `local-${Date.now()}`,
      session_id: activeSession?.id || '',
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMessage])

    try {
      const assistantId = `assistant-${Date.now()}`
      let started = false
      let newSessionId = activeSession?.id
      let pendingSources: Source[] = []

      await askQuestionStream(trimmed, courseId, activeSession?.id, userId, {
        onSession: (id) => {
          newSessionId = id
        },
        onSources: (s) => {
          pendingSources = s
        },
        onToken: (delta) => {
          if (!started) {
            started = true
            setIsTyping(false)
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                session_id: newSessionId || '',
                role: 'assistant',
                content: delta,
                sources: pendingSources,
                timestamp: new Date().toISOString(),
              },
            ])
          } else {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m)),
            )
          }
        },
        onDone: (id) => {
          if (id) newSessionId = id
        },
      })

      if (!activeSession && newSessionId) {
        await loadSessions()
        const refreshed = await getChatSessions(userId)
        const newSess = refreshed.find((s) => s.id === newSessionId)
        if (newSess) setActiveSession(newSess)
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          session_id: activeSession?.id || '',
          role: 'assistant',
          content: 'Sorry, I hit a snag. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const handleDeleteSession = async (session: ChatSession) => {
    try {
      await deleteSession(session.id)
      if (activeSession?.id === session.id) {
        setActiveSession(null)
        setMessages([])
      }
      await loadSessions()
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }

  const startNew = () => {
    setActiveSession(null)
    setMessages([])
    setHistoryOpen(false)
    textareaRef.current?.focus()
  }

  const isEmpty = messages.length === 0 && !isTyping
  const hasFiles = !!files && files.length > 0
  const lastMsg = messages[messages.length - 1]
  const showFollowups = !isTyping && lastMsg?.role === 'assistant'

  const composer = (variant: 'hero' | 'docked') => (
    <form
      onSubmit={handleAsk}
      className={cn(
        'relative flex w-full items-end gap-2 rounded-[20px] border border-white/12 bg-white/[0.03] p-2.5 shadow-lg transition-all',
        'focus-within:border-cyan-400/60 focus-within:bg-white/[0.05] focus-within:glow-brand-sm',
      )}
    >
      <textarea
        ref={textareaRef}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={variant === 'hero' ? 2 : 1}
        autoFocus={variant === 'hero'}
        placeholder={hasFiles ? 'Ask anything about your course…' : 'Ask a question…'}
        className="max-h-[200px] flex-1 resize-none bg-transparent px-2.5 py-2 text-[15px] text-zinc-100 placeholder-zinc-500 outline-none"
      />
      <Button
        type="submit"
        variant="primary"
        loading={isTyping}
        disabled={isTyping || !question.trim()}
        className="!h-10 !w-10 flex-shrink-0 !rounded-xl !p-0"
        aria-label="Send message"
      >
        {!isTyping && <ArrowUp className="h-[18px] w-[18px]" />}
      </Button>
    </form>
  )

  return (
    <div className="relative flex h-full flex-col">
      {/* Floating controls (no second header bar — the Learn mode bar is the only one) */}
      <div className="absolute right-4 top-3 z-20 flex items-center gap-2">
        <button
          onClick={() => setHistoryOpen(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[13px] text-zinc-300 backdrop-blur transition-colors hover:bg-white/[0.08] hover:text-zinc-100"
        >
          <History className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">History</span>
          {sessions.length > 0 && (
            <span className="rounded-full bg-white/[0.1] px-1.5 text-[11px] text-zinc-400">{sessions.length}</span>
          )}
        </button>
        {!isEmpty && (
          <button
            onClick={startNew}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[13px] text-zinc-300 backdrop-blur transition-colors hover:bg-white/[0.08] hover:text-zinc-100"
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New</span>
          </button>
        )}
      </div>

      {isEmpty ? (
        /* ── Center-first new-chat state ─────────────────────────── */
        <div className="flex flex-1 flex-col items-center justify-center px-4 pb-10">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-2xl"
          >
            <div className="mb-6 text-center">
              <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand" />
              <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
                What do you want to understand?
              </h1>
              <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
                {hasFiles
                  ? 'Ask anything — answers are grounded in your own materials, with exact page citations.'
                  : 'Upload course files from Materials to get answers grounded in your own notes.'}
              </p>
              {hasFiles && (
                <div className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-zinc-400">
                  <GraduationCap className="h-3.5 w-3.5 text-cyan-400/80" />
                  {course?.title ?? 'Course'} · {files!.length} sources indexed
                </div>
              )}
            </div>

            {composer('hero')}

            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {STARTERS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => handleAsk(undefined, s.prompt)}
                  disabled={!courseId}
                  className="rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-zinc-300 transition-all hover:border-cyan-400/40 hover:bg-white/[0.06] hover:text-zinc-100 disabled:opacity-50"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      ) : (
        /* ── Conversation ────────────────────────────────────────── */
        <>
          <div className="flex-1 overflow-y-auto">
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
                      <div className="max-w-[80%] rounded-2xl rounded-br-md border border-cyan-400/20 bg-cyan-500/[0.12] px-4 py-2.5 text-[15px] leading-relaxed text-zinc-50">
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    ) : (
                      <div className="min-w-0">
                        <Markdown content={msg.content} className="text-[15px] leading-relaxed text-zinc-200" />
                        {msg.sources && msg.sources.length > 0 && <Sources sources={msg.sources} />}
                        <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                          <CopyButton text={msg.content} />
                          <span className="text-[11px] text-zinc-600">{formatTime(msg.timestamp)}</span>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>

              {isTyping && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-8 flex items-center gap-1.5">
                  <span className="text-sm text-zinc-400">Searching your materials</span>
                  <span className="flex gap-1">
                    {[0, 0.15, 0.3].map((delay) => (
                      <span
                        key={delay}
                        className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400/80"
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
                  {FOLLOWUPS.map(({ icon: Icon, label, prompt }) => (
                    <button
                      key={label}
                      onClick={() => handleAsk(undefined, prompt)}
                      className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[13px] text-zinc-300 transition-all hover:border-cyan-400/40 hover:bg-white/[0.06] hover:text-zinc-100"
                    >
                      <Icon className="h-3.5 w-3.5 text-cyan-400/80" />
                      {label}
                    </button>
                  ))}
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Docked composer */}
          {courseId && (
            <div className="flex-shrink-0 px-4 pb-4 pt-2">
              <div className="mx-auto max-w-3xl">
                {composer('docked')}
                <p className="mt-2 text-center text-[11px] text-zinc-600">
                  Grounded in your {course?.title ?? 'course'} materials · Enter to send, Shift+Enter for a new line
                </p>
              </div>
            </div>
          )}
        </>
      )}

      {/* History slide-over */}
      <AnimatePresence>
        {historyOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setHistoryOpen(false)}
              className="absolute inset-0 z-30 bg-black/50 backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
              className="absolute inset-y-0 left-0 z-40 flex w-[320px] flex-col border-r border-white/10 bg-[#0c0f18]"
            >
              <div className="flex h-14 items-center justify-between px-4">
                <span className="text-sm font-semibold text-zinc-100">Chat history</span>
                <button
                  onClick={() => setHistoryOpen(false)}
                  className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
                  aria-label="Close history"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="px-3 pb-3">
                <Button variant="secondary" onClick={startNew} leftIcon={<Plus className="h-4 w-4" />} className="w-full">
                  New chat
                </Button>
              </div>
              <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
                {sessions.length === 0 ? (
                  <p className="px-2 py-6 text-center text-xs text-zinc-600">No conversations yet.</p>
                ) : (
                  sessions.map((session) => {
                    const active = activeSession?.id === session.id
                    return (
                      <div
                        key={session.id}
                        onClick={() => {
                          setActiveSession(session)
                          setHistoryOpen(false)
                        }}
                        className={cn(
                          'group flex cursor-pointer items-center justify-between gap-2 rounded-lg border px-2.5 py-2 transition-colors',
                          active
                            ? 'border-cyan-400/25 bg-cyan-500/[0.08] text-zinc-100'
                            : 'border-transparent text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200',
                        )}
                      >
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{session.title || 'Untitled chat'}</p>
                          <p className="text-[11px] text-zinc-600">
                            {new Date(session.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteSession(session)
                          }}
                          className="p-1 text-zinc-600 opacity-0 transition-all hover:text-rose-400 group-hover:opacity-100"
                          aria-label="Delete chat"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    )
                  })
                )}
              </div>
              <div className="border-t border-white/10 p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <GraduationCap className="h-3.5 w-3.5" />
                  {course?.title ?? 'Course'} · {hasFiles ? `${files!.length} files` : 'no files'}
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
