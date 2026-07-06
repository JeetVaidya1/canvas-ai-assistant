import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { BrandMark } from '@/components/ui/BrandMark'
import ExamMode from '@/components/ExamMode'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'

export default function ExamsPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'exams')
  }, [courseId])

  if (!courseId) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center px-6">
        <BrandMark className="mb-5 h-14 w-14" />
        <h3 className="text-lg font-semibold text-ink mb-1.5">Select a course first</h3>
        <p className="text-sm text-ink-soft max-w-sm">Pick a course from the sidebar to use Exam Mode.</p>
      </div>
    )
  }

  // ExamMode owns its own layout (center-first setup, focused live column with a
  // sticky bar + slide-over navigator, report-style results), so give it the full
  // bleed instead of a constraining max-width wrapper.
  return (
    <div className="h-full">
      <ExamMode courseId={courseId} userId={userId} />
    </div>
  )
}
