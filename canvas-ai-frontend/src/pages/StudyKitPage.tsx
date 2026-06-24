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
      <div className="px-6 pt-5 pb-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand">
            Study Kit
          </p>
          <h1 className="text-lg font-semibold text-zinc-100">{title}</h1>
        </div>
        <SubTabs
          tabs={TABS}
          active={tab}
          onChange={(key) => setTab(key as StudyKitTab)}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === 'notes' ? (
          <div className="p-5 max-w-4xl mx-auto" data-user-id={userId}>
            <NotesCreator courseId={courseId || ''} courseName={title} />
          </div>
        ) : (
          <AudioPage />
        )}
      </div>
    </div>
  )
}
