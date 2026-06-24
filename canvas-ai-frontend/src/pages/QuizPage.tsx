import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import QuizMode from '@/components/QuizMode'
import QuizAssistant from '@/components/QuizAsisstant'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'

type QuizTab = 'quiz' | 'helper'

export default function QuizPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const [tab, setTab] = useState<QuizTab>('quiz')

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'quiz')
  }, [courseId])

  return (
    <div className="py-5">
      <div className="max-w-3xl mx-auto px-5 mb-2">
        <div className="inline-flex bg-zinc-800/60 border border-zinc-700/40 rounded-lg p-1 gap-1">
          <button
            onClick={() => setTab('quiz')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              tab === 'quiz'
                ? 'bg-gradient-brand text-white glow-brand-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Quiz
          </button>
          <button
            onClick={() => setTab('helper')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              tab === 'helper'
                ? 'bg-gradient-brand text-white glow-brand-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Answer Helper
          </button>
        </div>
      </div>

      {tab === 'quiz' ? (
        <QuizMode courseId={courseId || ''} userId={userId} />
      ) : (
        <div className="max-w-3xl mx-auto px-5">
          <QuizAssistant courseId={courseId || ''} />
        </div>
      )}
    </div>
  )
}
