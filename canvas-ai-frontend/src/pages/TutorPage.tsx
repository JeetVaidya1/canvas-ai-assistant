import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import SocraticTutor from '@/components/SocraticTutor'
import { trackVisit } from '@/hooks/useRecentActivity'

type TutorTab = 'socratic' | 'feynman'

export default function TutorPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const [tab, setTab] = useState<TutorTab>('socratic')

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'tutor')
  }, [courseId])

  return (
    <div className="max-w-3xl mx-auto px-5 py-5">
      <div className="inline-flex bg-zinc-800/60 border border-zinc-700/40 rounded-lg p-1 gap-1 mb-3">
        <button
          onClick={() => setTab('socratic')}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            tab === 'socratic' ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Socratic
        </button>
        <button
          onClick={() => setTab('feynman')}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            tab === 'feynman' ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Feynman
        </button>
      </div>

      {tab === 'socratic' ? (
        <SocraticTutor courseId={courseId || ''} />
      ) : (
        <div className="text-center py-16 text-zinc-500 text-sm">Feynman mode coming up next.</div>
      )}
    </div>
  )
}
