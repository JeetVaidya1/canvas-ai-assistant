import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import NotesCreator from '@/components/NotesCreator'
import { useCourses } from '@/hooks/useCourses'
import { trackVisit } from '@/hooks/useRecentActivity'

export default function NotesPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const { data: courses } = useCourses()
  const course = courses?.find((c) => c.course_id === courseId)

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'notes')
  }, [courseId])

  return (
    <div className="p-5 max-w-4xl mx-auto">
      <NotesCreator courseId={courseId || ''} courseName={course?.title || ''} />
    </div>
  )
}
