import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { MessagesSquare, Lightbulb, PenLine } from 'lucide-react'
import ChatPage from '@/pages/ChatPage'
import SocraticTutor from '@/components/SocraticTutor'
import FeynmanMode from '@/components/FeynmanMode'
import { SubTabs, type SubTab } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'

type LearnTab = 'conversation' | 'socratic' | 'feynman'

const TABS: SubTab[] = [
  { key: 'conversation', label: 'Conversation', icon: <MessagesSquare className="w-4 h-4" /> },
  { key: 'socratic', label: 'Socratic Tutor', icon: <Lightbulb className="w-4 h-4" /> },
  { key: 'feynman', label: 'Feynman', icon: <PenLine className="w-4 h-4" /> },
]

export default function LearnPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const [tab, setTab] = useState<LearnTab>('conversation')

  const course = courses?.find((c) => c.course_id === courseId)
  const title = course?.title ?? 'Learn'
  const id = courseId ?? ''

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-5 pb-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand">Learn</p>
          <h1 className="text-lg font-semibold text-zinc-100">{title}</h1>
        </div>
        <SubTabs tabs={TABS} active={tab} onChange={(key) => setTab(key as LearnTab)} />
      </div>
      <div className="flex-1 min-h-0 overflow-hidden">
        {tab === 'conversation' ? (
          <ChatPage />
        ) : tab === 'socratic' ? (
          <SocraticTutor courseId={id} />
        ) : (
          <FeynmanMode courseId={id} userId={userId} />
        )}
      </div>
    </div>
  )
}
