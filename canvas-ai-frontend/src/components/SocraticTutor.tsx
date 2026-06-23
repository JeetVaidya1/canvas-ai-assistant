import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, User, Bot, Lightbulb } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { socraticTurn, type TutorTurn } from '@/lib/api'

interface SocraticTutorProps {
  courseId: string
}

export default function SocraticTutor({ courseId }: SocraticTutorProps) {
  const [turns, setTurns] = useState<TutorTurn[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, thinking])

  const send = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    const msg = input.trim()
    if (!msg || !courseId || thinking) return
    setInput('')
    const history = turns
    setTurns((prev) => [...prev, { role: 'user', content: msg }])
    setThinking(true)
    try {
      const { reply } = await socraticTurn(courseId, msg, history)
      setTurns((prev) => [...prev, { role: 'assistant', content: reply }])
    } catch {
      setTurns((prev) => [...prev, { role: 'assistant', content: 'Sorry — I hit a snag. Try again.' }])
    } finally {
      setThinking(false)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-9rem)]">
      <div className="flex-1 overflow-y-auto p-2 space-y-5">
        {turns.length === 0 && !thinking ? (
          <div className="text-center py-16">
            <div className="w-14 h-14 rounded-full bg-violet-500/10 flex items-center justify-center mx-auto mb-4">
              <Lightbulb className="w-7 h-7 text-violet-400" />
            </div>
            <h3 className="text-lg font-semibold text-zinc-100 mb-2">Socratic tutor</h3>
            <p className="text-sm text-zinc-500 max-w-md mx-auto mb-6">
              I won't just give you the answer — I'll ask questions, grounded in your course, until it clicks.
              Tell me what you're working on or stuck on.
            </p>
            <div className="flex flex-wrap justify-center gap-2 max-w-xl mx-auto">
              {['Walk me through this problem', 'Help me understand this proof', 'I keep getting this wrong'].map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1.5 rounded-lg text-sm border border-zinc-700"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map((t, i) => (
            <div key={i} className={`flex items-start gap-3 ${t.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${t.role === 'user' ? 'bg-zinc-700' : 'bg-violet-500/15'}`}>
                {t.role === 'user' ? <User className="w-3.5 h-3.5 text-zinc-300" /> : <Bot className="w-3.5 h-3.5 text-violet-400" />}
              </div>
              <div className={`max-w-2xl ${t.role === 'user' ? 'text-right' : ''}`}>
                <div className={`inline-block px-4 py-3 rounded-lg ${t.role === 'user' ? 'bg-cyan-600 text-white' : 'bg-zinc-800 text-zinc-200 border border-zinc-700'}`}>
                  {t.role === 'user' ? (
                    <p className="whitespace-pre-wrap leading-relaxed text-sm">{t.content}</p>
                  ) : (
                    <Markdown content={t.content} className="text-sm" />
                  )}
                </div>
              </div>
            </div>
          ))
        )}
        {thinking && (
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-full bg-violet-500/15 flex items-center justify-center">
              <Bot className="w-3.5 h-3.5 text-violet-400" />
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
        <div ref={endRef} />
      </div>

      <form onSubmit={send} className="flex gap-3 pt-3 border-t border-zinc-800">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="What are you working on?"
          disabled={thinking}
          className="flex-1 px-4 py-2.5 bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-zinc-600 outline-none text-sm"
        />
        <button
          type="submit"
          disabled={thinking || !input.trim()}
          className="bg-violet-600 hover:bg-violet-500 text-white p-2.5 rounded-lg disabled:opacity-50 transition-colors"
        >
          {thinking ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
        </button>
      </form>
    </div>
  )
}
