import { useState, useRef, useEffect } from 'react'
import { Send, User, Bot, Lightbulb } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
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
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      <div className="flex-1 overflow-y-auto p-2 space-y-5">
        {turns.length === 0 && !thinking ? (
          <div className="text-center py-16">
            <div className="w-14 h-14 rounded-2xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center mx-auto mb-5">
              <Lightbulb className="w-7 h-7 text-cyan-300" />
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
                  className="bg-zinc-800/70 hover:bg-zinc-700/80 text-zinc-300 px-3 py-1.5 rounded-lg text-sm border border-zinc-700 hover:border-cyan-500/30 transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map((t, i) => (
            <div key={i} className={`flex items-start gap-3 ${t.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${t.role === 'user' ? 'bg-zinc-700' : 'bg-gradient-brand-soft border border-cyan-500/20'}`}>
                {t.role === 'user' ? <User className="w-3.5 h-3.5 text-zinc-300" /> : <Bot className="w-3.5 h-3.5 text-cyan-300" />}
              </div>
              <div className={`max-w-2xl ${t.role === 'user' ? 'text-right' : ''}`}>
                <div className={`inline-block px-4 py-3 rounded-xl ${t.role === 'user' ? 'bg-gradient-brand text-white glow-brand-sm' : 'card-surface text-zinc-200'}`}>
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
            <div className="w-7 h-7 rounded-full bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center">
              <Bot className="w-3.5 h-3.5 text-cyan-300" />
            </div>
            <div className="card-surface rounded-xl px-4 py-3">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-cyan-400/70 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-cyan-400/70 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                <div className="w-2 h-2 bg-cyan-400/70 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
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
          className="flex-1 px-4 py-2.5 bg-zinc-800/70 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-sm transition-colors"
        />
        <Button
          type="submit"
          loading={thinking}
          disabled={thinking || !input.trim()}
          aria-label="Send message"
          className="px-3"
        >
          {!thinking && <Send className="w-5 h-5" />}
        </Button>
      </form>
    </div>
  )
}
