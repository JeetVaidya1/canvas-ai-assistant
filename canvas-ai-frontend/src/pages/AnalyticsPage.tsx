import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import AnalyticsDashboard from '@/components/AnalyticsDashboard'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'

export default function AnalyticsPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'analytics')
  }, [courseId])

  return (
    <div className="p-5 max-w-5xl mx-auto">
      <AnalyticsDashboard courseId={courseId || ''} userId={userId} />
    </div>
  )
}
