import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { motion } from 'motion/react'
import { Plus, History, GraduationCap } from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { Composer } from '@/components/learn/Composer'
import { MessageList } from '@/components/learn/MessageList'
import { HistoryDrawer } from '@/components/learn/HistoryDrawer'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { trackVisit } from '@/hooks/useRecentActivity'
import { useSessions, useSessionMessages, useDeleteSession } from '@/hooks/useChatSessions'
import ErrorInline from '@/components/shared/ErrorInline'
import { showError } from '@/lib/toast'
import {
  askQuestionStream,
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

export default function ChatPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const { data: files } = useCourseFiles(courseId)

  const course = courses?.find((c) => c.course_id === courseId)

  const qc = useQueryClient()
  const sessionsQuery = useSessions(userId)
  const sessions = sessionsQuery.data ?? []

  const [activeSession, setActiveSession] = useState<ChatSession | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // Tracks an in-flight SSE stream so cached history never clobbers it.
  const streamingRef = useRef(false)

  const messagesQuery = useSessionMessages(activeSession?.id)
  const deleteSessionMutation = useDeleteSession(userId)

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'chat')
  }, [courseId])

  // Sync cached/fetched history into the local, streaming-aware message list.
  useEffect(() => {
    if (!activeSession || streamingRef.current) return
    if (messagesQuery.data) setMessages(messagesQuery.data)
  }, [activeSession, messagesQuery.data])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const handleAsk = async (presetText?: string) => {
    const trimmed = (presetText ?? question).trim()
    if (!trimmed || !courseId || isTyping) return

    setQuestion('')
    setIsTyping(true)
    streamingRef.current = true

    const userMessage: Message = {
      id: `local-${Date.now()}`,
      session_id: activeSession?.id || '',
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    }
    // `messages` is current for this render — the base of the new exchange.
    const baseMessages = [...messages, userMessage]
    setMessages(baseMessages)

    try {
      const assistantId = `assistant-${Date.now()}`
      let started = false
      let newSessionId = activeSession?.id
      let pendingSources: Source[] = []
      let assistantContent = ''
      let assistantTimestamp = ''

      await askQuestionStream(trimmed, courseId, activeSession?.id, userId, {
        onSession: (id) => {
          newSessionId = id
        },
        onSources: (s) => {
          pendingSources = s
        },
        onToken: (delta) => {
          assistantContent += delta
          if (!started) {
            started = true
            assistantTimestamp = new Date().toISOString()
            setIsTyping(false)
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                session_id: newSessionId || '',
                role: 'assistant',
                content: delta,
                sources: pendingSources,
                timestamp: assistantTimestamp,
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

      if (newSessionId) {
        // Seed the messages cache with the completed exchange so selecting
        // this session doesn't refetch (or momentarily blank) the transcript.
        const finalMessages: Message[] = started
          ? [
              ...baseMessages,
              {
                id: assistantId,
                session_id: newSessionId,
                role: 'assistant',
                content: assistantContent,
                sources: pendingSources,
                timestamp: assistantTimestamp,
              },
            ]
          : baseMessages
        qc.setQueryData<Message[]>(['messages', newSessionId], finalMessages)
      }
      if (!activeSession && newSessionId) {
        // Single refresh of the session list (was two sequential fetches),
        // then adopt the new session straight from the cache.
        await qc.invalidateQueries({ queryKey: ['sessions', userId] })
        const refreshed = qc.getQueryData<ChatSession[]>(['sessions', userId])
        const newSess = refreshed?.find((s) => s.id === newSessionId)
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
      streamingRef.current = false
    }
  }

  const handleDeleteSession = async (session: ChatSession) => {
    try {
      await deleteSessionMutation.mutateAsync(session.id)
      if (activeSession?.id === session.id) {
        setActiveSession(null)
        setMessages([])
      }
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to delete chat')
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

      {isEmpty && activeSession && messagesQuery.isError ? (
        /* ── Conversation failed to load ─────────────────────────── */
        <div className="flex flex-1 items-center justify-center px-4">
          <ErrorInline
            message="Couldn't load this conversation."
            onRetry={() => void messagesQuery.refetch()}
            className="w-full max-w-md"
          />
        </div>
      ) : isEmpty ? (
        /* ── Center-first new-chat state ─────────────────────────── */
        <div className="flex flex-1 flex-col items-center justify-center px-4 pb-10">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
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

            <Composer
              variant="hero"
              value={question}
              onChange={setQuestion}
              onSend={() => void handleAsk()}
              sending={isTyping}
              hasFiles={hasFiles}
              textareaRef={textareaRef}
            />

            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {STARTERS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => void handleAsk(s.prompt)}
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
            <MessageList
              messages={messages}
              isTyping={isTyping}
              showFollowups={showFollowups}
              onFollowup={(prompt) => void handleAsk(prompt)}
              bottomRef={messagesEndRef}
            />
          </div>

          {/* Docked composer */}
          {courseId && (
            <div className="flex-shrink-0 px-4 pb-4 pt-2">
              <div className="mx-auto max-w-3xl">
                <Composer
                  variant="docked"
                  value={question}
                  onChange={setQuestion}
                  onSend={() => void handleAsk()}
                  sending={isTyping}
                  hasFiles={hasFiles}
                  textareaRef={textareaRef}
                />
                <p className="mt-2 text-center text-[11px] text-zinc-500">
                  Grounded in your {course?.title ?? 'course'} materials · Enter to send, Shift+Enter for a new line
                </p>
              </div>
            </div>
          )}
        </>
      )}

      <HistoryDrawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        sessions={sessions}
        loadFailed={sessionsQuery.isError}
        onRetryLoad={() => void sessionsQuery.refetch()}
        activeSessionId={activeSession?.id}
        onSelect={(session) => {
          setActiveSession(session)
          setHistoryOpen(false)
        }}
        onDelete={(session) => void handleDeleteSession(session)}
        onNewChat={startNew}
        courseTitle={course?.title ?? 'Course'}
        fileCount={files?.length ?? 0}
      />
    </div>
  )
}
