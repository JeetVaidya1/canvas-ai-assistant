import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import {
  Send,
  Loader2,
  User,
  Plus,
  Trash2,
  MessageCircle,
  CheckCircle,
  Bot,
  FileText,
} from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
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
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'chat')
  }, [courseId])

  useEffect(() => {
    loadSessions()
  }, [userId])

  useEffect(() => {
    if (activeSession) loadMessages()
  }, [activeSession])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const loadSessions = async () => {
    try {
      const data = await getChatSessions(userId)
      setSessions(data || [])
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }

  const loadMessages = async () => {
    if (!activeSession) return
    try {
      const data = await getSessionMessages(activeSession.id)
      setMessages(data || [])
    } catch (e) {
      console.error('Failed to load messages:', e)
    }
  }

  const handleAsk = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!question.trim() || !courseId) return

    const userQuestion = question
    setQuestion('')
    setIsTyping(true)

    const userMessage: Message = {
      id: `local-${Date.now()}`,
      session_id: activeSession?.id || '',
      role: 'user',
      content: userQuestion,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMessage])

    try {
      const assistantId = `assistant-${Date.now()}`
      let started = false
      let newSessionId = activeSession?.id
      let pendingSources: Source[] = []

      await askQuestionStream(userQuestion, courseId, activeSession?.id, userId, {
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
  }

  return (
    <div className="flex h-full">
      {/* Sessions sidebar */}
      {sidebarOpen && (
        <div className="w-60 bg-zinc-950 border-r border-zinc-800 flex flex-col">
          <div className="p-4 border-b border-zinc-800">
            <button
              onClick={startNew}
              className="w-full bg-zinc-800 hover:bg-zinc-700 text-zinc-200 py-2.5 px-4 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 text-sm"
            >
              <Plus className="w-4 h-4" /> New Chat
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`group flex items-center justify-between p-2.5 rounded-lg cursor-pointer transition-colors ${
                  activeSession?.id === session.id
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300'
                }`}
                onClick={() => setActiveSession(session)}
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{session.title || 'Untitled Chat'}</p>
                  <p className="text-xs text-zinc-600">{new Date(session.created_at).toLocaleDateString()}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteSession(session)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 text-zinc-500 hover:text-red-400 transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="h-12 flex items-center px-4 border-b border-zinc-800 gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 hover:bg-zinc-800 rounded-lg transition-colors text-zinc-400"
          >
            <MessageCircle className="w-4 h-4" />
          </button>
          <span className="text-sm font-medium text-zinc-200">
            {course?.title ?? 'Study Chat'}
          </span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && !isTyping ? (
            <div className="text-center py-16">
              <div className="w-14 h-14 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-5">
                <MessageCircle className="w-7 h-7 text-zinc-500" />
              </div>
              <h3 className="text-lg font-semibold text-zinc-100 mb-2">Ready to help you study</h3>
              <p className="text-sm text-zinc-500 max-w-md mx-auto mb-8">
                Ask me anything about your course materials.
              </p>
              <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
                {[
                  'Summarize the key concepts',
                  'Quiz me on this topic',
                  'Explain chapter 3',
                  'What are the main themes?',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setQuestion(suggestion)}
                    className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-4 py-2 rounded-lg text-sm transition-colors border border-zinc-700"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex items-start gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                      msg.role === 'user'
                        ? 'bg-zinc-700'
                        : 'bg-zinc-800'
                    }`}
                  >
                    {msg.role === 'user' ? (
                      <User className="w-3.5 h-3.5 text-zinc-300" />
                    ) : (
                      <Bot className="w-3.5 h-3.5 text-cyan-400" />
                    )}
                  </div>
                  <div className={`max-w-3xl ${msg.role === 'user' ? 'text-right' : ''}`}>
                    <div
                      className={`inline-block px-4 py-3 rounded-lg ${
                        msg.role === 'user'
                          ? 'bg-cyan-600 text-white'
                          : 'bg-zinc-800 text-zinc-200 border border-zinc-700'
                      }`}
                    >
                      {msg.role === 'user' ? (
                        <p className="whitespace-pre-wrap leading-relaxed text-sm">{msg.content}</p>
                      ) : (
                        <Markdown content={msg.content} className="text-sm" />
                      )}
                    </div>
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {msg.sources.map((s: Source, i: number) => (
                          <span
                            key={i}
                            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-800/60 px-2 py-0.5 text-xs text-zinc-400"
                            title={s.file}
                          >
                            <FileText className="w-3 h-3 text-cyan-400/70" />
                            <span className="max-w-[200px] truncate">{s.file}</span>
                            {s.page ? <span className="text-zinc-500">p.{s.page}</span> : null}
                          </span>
                        ))}
                      </div>
                    )}
                    <p className="text-xs text-zinc-600 mt-1 px-1">
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex items-start gap-3">
                  <div className="w-7 h-7 rounded-full bg-zinc-800 flex items-center justify-center">
                    <Bot className="w-3.5 h-3.5 text-cyan-400" />
                  </div>
                  <div className="bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        {courseId && files && files.length > 0 && (
          <div className="border-t border-zinc-800 p-4 bg-zinc-950">
            <form onSubmit={handleAsk} className="flex gap-3">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask me anything about your course..."
                className="flex-1 px-4 py-2.5 bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none text-sm"
                disabled={isTyping}
              />
              <button
                type="submit"
                disabled={isTyping || !question.trim()}
                className="bg-cyan-600 hover:bg-cyan-500 text-white p-2.5 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isTyping ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </form>
            <div className="flex items-center gap-2 mt-3 text-xs text-zinc-600">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
              <span>Knowledge base: {files.length} files from {course?.title}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
