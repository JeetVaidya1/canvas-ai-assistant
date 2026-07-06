import { useParams } from 'react-router-dom'
import NotesCreator from '@/components/NotesCreator'
import { useCourses } from '@/hooks/useCourses'

/**
 * "Study Kit" destination — grounded notes plus the flashcards generated from
 * them (flashcards render inside NotesCreator, decomposed under
 * components/studykit/). Audio overviews were removed until the feature
 * actually exists; no "coming soon" dead ends.
 */
export default function StudyKitPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const { data: courses } = useCourses()

  const course = courses?.find((c) => c.course_id === courseId)
  const title = course?.title || 'Study Kit'

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 h-14 border-b border-line flex items-center flex-shrink-0">
        <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-faint pl-1.5">
          Grounded notes & flashcards
        </span>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="h-full px-5 py-4 max-w-4xl mx-auto">
          <NotesCreator courseId={courseId || ''} courseName={title} />
        </div>
      </div>
    </div>
  )
}
