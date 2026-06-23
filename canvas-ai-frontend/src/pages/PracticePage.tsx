import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import PracticeMode from '@/components/PracticeMode'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'

export default function PracticePage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'practice')
  }, [courseId])

  return (
    <div className="p-5 max-w-3xl mx-auto">
      <PracticeMode courseId={courseId || ''} userId={userId} />
    </div>
  )
}
