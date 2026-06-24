import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { FileText, Headphones } from 'lucide-react'
import { SubTabs } from '@/components/ui/SubTabs'
import type { SubTab } from '@/components/ui/SubTabs'
import NotesCreator from '@/components/NotesCreator'
import AudioPage from '@/pages/AudioPage'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'

type StudyKitTab = 'notes' | 'audio'

const TABS: SubTab[] = [
  { key: 'notes', label: 'Notes', icon: <FileText className="w-4 h-4" /> },
  { key: 'audio', label: 'Audio', icon: <Headphones className="w-4 h-4" /> },
]

/**
 * Consolidated "Study Kit" destination that composes the existing Notes and
 * Audio experiences behind a segmented control. Reuses NotesCreator (which
 * includes generated flashcards internally) and AudioPage as-is.
 */
export default function StudyKitPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const [tab, setTab] = useState<StudyKitTab>('notes')

  const course = courses?.find((c) => c.course_id === courseId)
  const title = course?.title || 'Study Kit'

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 h-14 border-b border-[#18181d] flex items-center justify-between flex-shrink-0 gap-4">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 pl-1.5">
          {tab === 'notes' ? 'Grounded notes & flashcards' : 'Listen on the go'}
        </span>
        <SubTabs
          tabs={TABS}
          active={tab}
          onChange={(key) => setTab(key as StudyKitTab)}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === 'notes' ? (
          <div className="h-full px-5 py-4 max-w-4xl mx-auto" data-user-id={userId}>
            <NotesCreator courseId={courseId || ''} courseName={title} />
          </div>
        ) : (
          <AudioPage />
        )}
      </div>
    </div>
  )
}
