import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { GraduationCap } from 'lucide-react'
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
        <div className="w-14 h-14 rounded-2xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center mb-5">
          <GraduationCap className="w-7 h-7 text-cyan-300" />
        </div>
        <h3 className="text-lg font-semibold text-zinc-100 mb-1.5">Select a course first</h3>
        <p className="text-sm text-zinc-500 max-w-sm">Pick a course from the sidebar to use Exam Mode.</p>
      </div>
    )
  }

  return (
    <div className="p-5 max-w-3xl mx-auto">
      <ExamMode courseId={courseId} userId={userId} />
    </div>
  )
}
