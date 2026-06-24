import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'motion/react'
import {
  ArrowUp,
  Plus,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  FileText,
  Sparkles,
  BookOpen,
  Library,
  ListChecks,
  Lightbulb,
} from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
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

const STARTERS: ReadonlyArray<{ icon: typeof BookOpen; label: string; prompt: string }> = [
  { icon: Library, label: 'Summarize', prompt: 'Summarize the key concepts from my course materials.' },
  { icon: ListChecks, label: 'Quiz me', prompt: 'Quiz me on the most important topics in this course.' },
  { icon: BookOpen, label: 'Explain', prompt: 'Explain the hardest concept in simple terms, with an example.' },
  { icon: Lightbulb, label: 'Key themes', prompt: 'What are the main themes and how do they connect?' },
]

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function fileLabel(file: string): string {
  const parts = file.split(/[/\\]/)
  return parts[parts.length - 1] || file
}

/** Elegant grouped source citations under an assistant answer. */
function SourceChips({ sources }: { sources: ReadonlyArray<Source> }) {
  if (sources.length === 0) return null
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05, duration: 0.25 }}
      className="mt-3 flex flex-wrap items-center gap-1.5"
    >
      <span className="mr-0.5 inline-flex items-center gap-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        <FileText className="h-3 w-3 text-cyan-400/80" />
        Grounded in
      </span>
      {sources.map((s, i) => (
        <span
          key={`${s.file}-${s.page ?? 'x'}-${i}`}
          title={s.file}
          className="inline-flex max-w-full items-center gap-1 rounded-md border border-cyan-500/20 bg-gradient-brand-soft px-2 py-0.5 text-xs text-zinc-300"
        >
          <span className="truncate max-w-[180px]">{fileLabel(s.file)}</span>
          {s.page != null && (
            <span className="rounded bg-cyan-500/15 px-1 text-[10px] font-semibold text-cyan-300">
              p.{s.page}
            </span>
          )}
        </span>
      ))}
    </motion.div>
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
  const [railOpen, setRailOpen] = useState(true)

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
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [question])

  const handleAsk = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    const trimmed = question.trim()
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
    textareaRef.current?.focus()
  }

  const isEmpty = messages.length === 0 && !isTyping
  const hasFiles = !!files && files.length > 0

  return (
    <div className="flex h-full bg-zinc-950">
      {/* Session rail */}
      <AnimatePresence initial={false}>
        {railOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 256, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="flex flex-col overflow-hidden border-r border-zinc-800/80 bg-zinc-950"
          >
            <div className="w-64 flex h-full flex-col">
              <div className="p-3">
                <Button
                  variant="secondary"
                  onClick={startNew}
                  leftIcon={<Plus className="h-4 w-4" />}
                  className="w-full"
                >
                  New chat
                </Button>
              </div>
              <div className="px-3 pb-1.5">
                <p className="text-[11px] font-semibold uppercase tracking-widest text-zinc-600">
                  History
                </p>
              </div>
              <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
                {sessions.length === 0 ? (
                  <p className="px-2 py-4 text-xs text-zinc-600">No conversations yet.</p>
                ) : (
                  sessions.map((session) => {
                    const active = activeSession?.id === session.id
                    return (
                      <div
                        key={session.id}
                        onClick={() => setActiveSession(session)}
                        className={`group flex cursor-pointer items-center justify-between rounded-lg border px-2.5 py-2 transition-colors ${
                          active
                            ? 'border-cyan-500/25 bg-gradient-brand-soft text-zinc-100'
                            : 'border-transparent text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                        }`}
                      >
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">
                            {session.title || 'Untitled chat'}
                          </p>
                          <p className="text-[11px] text-zinc-600">
                            {new Date(session.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteSession(session)
                          }}
                          className="p-1 text-zinc-600 opacity-0 transition-all hover:text-red-400 group-hover:opacity-100"
                          aria-label="Delete chat"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    )
                  })
                )}
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Conversation column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <header className="flex h-12 flex-shrink-0 items-center gap-2.5 border-b border-zinc-800/80 px-3">
          <button
            onClick={() => setRailOpen((v) => !v)}
            className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-cyan-300"
            aria-label={railOpen ? 'Hide history' : 'Show history'}
          >
            {railOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
          </button>
          <div className="flex h-6 w-6 items-center justify-center rounded-lg border border-cyan-500/20 bg-gradient-brand-soft">
            <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
          </div>
          <span className="truncate text-sm font-medium text-zinc-200">
            {course?.title ?? 'Study Chat'}
          </span>
          {hasFiles && (
            <span className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[11px] text-zinc-400">
              <Library className="h-3 w-3 text-cyan-400/80" />
              {files!.length} source{files!.length !== 1 ? 's' : ''}
            </span>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {isEmpty ? (
            <div className="flex h-full items-center justify-center px-6">
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, ease: 'easeOut' }}
                className="w-full max-w-2xl text-center"
              >
                <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-500/20 bg-gradient-brand-soft glow-brand">
                  <Sparkles className="h-8 w-8 text-cyan-300" />
                </div>
                <h2 className="text-2xl font-semibold tracking-tight">
                  <span className="text-gradient-brand">Ask your course</span>
                </h2>
                <p className="mx-auto mt-2 max-w-md text-sm text-zinc-500">
                  {hasFiles
                    ? 'Streamed answers grounded in your materials — with exact page citations and full conversation memory.'
                    : 'Upload course files from the overview page to get answers grounded in your own materials.'}
                </p>
                <div className="mx-auto mt-7 grid max-w-xl grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {STARTERS.map(({ icon: Icon, label, prompt }) => (
                    <button
                      key={label}
                      onClick={() => {
                        setQuestion(prompt)
                        textareaRef.current?.focus()
                      }}
                      className="card-interactive group flex items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3.5 text-left transition-colors hover:border-cyan-500/40"
                    >
                      <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border border-cyan-500/15 bg-gradient-brand-soft">
                        <Icon className="h-4 w-4 text-cyan-300" />
                      </span>
                      <span className="min-w-0">
                        <span className="block text-sm font-medium text-zinc-200 group-hover:text-zinc-100">
                          {label}
                        </span>
                        <span className="mt-0.5 line-clamp-2 block text-xs text-zinc-500">
                          {prompt}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            <div className="mx-auto w-full max-w-3xl space-y-7 px-6 py-8">
              <AnimatePresence initial={false}>
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.28, ease: 'easeOut' }}
                    className={msg.role === 'user' ? 'flex justify-end' : 'flex flex-col'}
                  >
                    {msg.role === 'user' ? (
                      <div className="max-w-[85%]">
                        <div className="rounded-2xl rounded-br-md border border-cyan-500/25 bg-cyan-500/10 px-4 py-2.5 text-sm leading-relaxed text-zinc-100">
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        </div>
                        <p className="mt-1 pr-1 text-right text-[11px] text-zinc-600">
                          {formatTime(msg.timestamp)}
                        </p>
                      </div>
                    ) : (
                      <div className="min-w-0">
                        <div className="mb-1.5 flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded-md border border-cyan-500/20 bg-gradient-brand-soft">
                            <Sparkles className="h-3 w-3 text-cyan-300" />
                          </span>
                          <span className="text-xs font-semibold text-zinc-400">Vindexa</span>
                          <span className="text-[11px] text-zinc-600">{formatTime(msg.timestamp)}</span>
                        </div>
                        <div className="pl-7">
                          <Markdown content={msg.content} className="text-sm leading-relaxed text-zinc-200" />
                          {msg.sources && msg.sources.length > 0 && (
                            <SourceChips sources={msg.sources} />
                          )}
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>

              {isTyping && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col"
                >
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-md border border-cyan-500/20 bg-gradient-brand-soft">
                      <Sparkles className="h-3 w-3 text-cyan-300" />
                    </span>
                    <span className="text-xs font-semibold text-zinc-400">Vindexa</span>
                  </div>
                  <div className="flex items-center gap-1.5 pl-7">
                    <span className="text-xs text-zinc-500">Searching your materials</span>
                    <span className="flex gap-1">
                      {[0, 0.15, 0.3].map((delay) => (
                        <span
                          key={delay}
                          className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400/80"
                          style={{ animationDelay: `${delay}s` }}
                        />
                      ))}
                    </span>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Composer */}
        {courseId && (
          <div className="flex-shrink-0 border-t border-zinc-800/80 bg-zinc-950 px-4 pb-4 pt-3">
            <div className="mx-auto max-w-3xl">
              <form
                onSubmit={handleAsk}
                className="flex items-end gap-2 rounded-2xl border border-zinc-700 bg-zinc-900/80 p-2 transition-colors focus-within:border-cyan-500/60 focus-within:glow-brand-sm"
              >
                <textarea
                  ref={textareaRef}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  placeholder={
                    hasFiles ? 'Ask anything about your course…' : 'Ask a question…'
                  }
                  className="max-h-[180px] flex-1 resize-none bg-transparent px-2.5 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none"
                />
                <Button
                  type="submit"
                  variant="primary"
                  loading={isTyping}
                  disabled={isTyping || !question.trim()}
                  className="!h-9 !w-9 flex-shrink-0 !p-0"
                  aria-label="Send message"
                >
                  {!isTyping && <ArrowUp className="h-4 w-4" />}
                </Button>
              </form>
              <p className="mt-2 px-1 text-center text-[11px] text-zinc-600">
                {hasFiles ? (
                  <>
                    Answers cite your {course?.title ?? 'course'} materials.{' '}
                    <span className="text-zinc-700">Press Enter to send, Shift+Enter for a new line.</span>
                  </>
                ) : (
                  'No materials uploaded yet — upload files from the course overview for grounded answers.'
                )}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
