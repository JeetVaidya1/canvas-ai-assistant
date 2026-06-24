import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { MessagesSquare, Lightbulb, PenLine } from 'lucide-react'
import ChatPage from '@/pages/ChatPage'
import SocraticTutor from '@/components/SocraticTutor'
import FeynmanMode from '@/components/FeynmanMode'
import { SubTabs, type SubTab } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'

type LearnTab = 'conversation' | 'socratic' | 'feynman'

const TABS: SubTab[] = [
  { key: 'conversation', label: 'Conversation', icon: <MessagesSquare className="w-4 h-4" /> },
  { key: 'socratic', label: 'Socratic Tutor', icon: <Lightbulb className="w-4 h-4" /> },
  { key: 'feynman', label: 'Feynman', icon: <PenLine className="w-4 h-4" /> },
]

export default function LearnPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const [tab, setTab] = useState<LearnTab>('conversation')

  const id = courseId ?? ''

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 h-14 border-b border-[#18181d] flex items-center justify-between flex-shrink-0">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 pl-1.5">
          {tab === 'conversation' ? 'Ask & explore' : tab === 'socratic' ? 'Be guided, not told' : 'Teach it back'}
        </span>
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
