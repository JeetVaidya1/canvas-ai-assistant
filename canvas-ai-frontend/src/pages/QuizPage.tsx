import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import QuizAssistant from '@/components/QuizAsisstant'
import { trackVisit } from '@/hooks/useRecentActivity'

export default function QuizPage() {
  const { courseId } = useParams<{ courseId: string }>()

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'quiz')
  }, [courseId])

  return (
    <div className="p-5 max-w-3xl mx-auto">
      <QuizAssistant courseId={courseId || ''} />
    </div>
  )
}
