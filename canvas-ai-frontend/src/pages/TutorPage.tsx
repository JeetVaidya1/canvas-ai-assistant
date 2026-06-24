import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { MessagesSquare, PenLine } from 'lucide-react'
import SocraticTutor from '@/components/SocraticTutor'
import FeynmanMode from '@/components/FeynmanMode'
import { PageHeader } from '@/components/ui/Card'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'

type TutorTab = 'socratic' | 'feynman'

const TABS: { id: TutorTab; label: string; icon: typeof MessagesSquare }[] = [
  { id: 'socratic', label: 'Socratic', icon: MessagesSquare },
  { id: 'feynman', label: 'Feynman', icon: PenLine },
]

export default function TutorPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const [tab, setTab] = useState<TutorTab>('socratic')

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'tutor')
  }, [courseId])

  return (
    <div className="max-w-3xl mx-auto px-5 py-6">
      <PageHeader
        eyebrow="AI Tutor"
        title="Tutor"
        subtitle="Learn by working through it — grounded in your course material."
        className="mb-5"
        actions={
          <div className="inline-flex bg-zinc-900/70 border border-zinc-800 rounded-xl p-1 gap-1">
            {TABS.map(({ id, label, icon: Icon }) => {
              const active = tab === id
              return (
                <button
                  key={id}
                  onClick={() => setTab(id)}
                  className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    active
                      ? 'bg-gradient-brand-soft text-cyan-300 ring-1 ring-cyan-400/30'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              )
            })}
          </div>
        }
      />

      {tab === 'socratic' ? (
        <SocraticTutor courseId={courseId || ''} />
      ) : (
        <FeynmanMode courseId={courseId || ''} userId={userId} />
      )}
    </div>
  )
}
