import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
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
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <h3 className="text-base font-medium text-zinc-300 mb-1">Select a course first</h3>
        <p className="text-sm text-zinc-500">Pick a course from the sidebar to use Exam Mode.</p>
      </div>
    )
  }

  return (
    <div className="p-5 max-w-3xl mx-auto">
      <ExamMode courseId={courseId} userId={userId} />
    </div>
  )
}
