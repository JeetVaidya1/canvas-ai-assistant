import { useParams } from 'react-router-dom'
import IntegrationsPanel from '@/components/IntegrationsPanel'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import CourseOverviewSkeleton from '@/components/skeletons/CourseOverviewSkeleton'
import UploadZone from '@/components/materials/UploadZone'
import FileList from '@/components/materials/FileList'
import { Card, PageHeader } from '@/components/ui/Card'

/**
 * Materials — the course's source of truth. Upload zone first (it powers
 * everything), then the indexed files, then integrations. Integrations are
 * always visible: Canvas import matters most when the course is still empty.
 */
export default function CourseOverview() {
  const { courseId } = useParams<{ courseId: string }>()
  const { data: courses } = useCourses()
  const { data: files } = useCourseFiles(courseId)

  const course = courses?.find((c) => c.course_id === courseId)
  if (!course || !courseId) return <CourseOverviewSkeleton />

  const fileCount = files?.length ?? 0
  const hasFiles = fileCount > 0

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <PageHeader
        eyebrow="Materials"
        title={course.title}
        subtitle={
          hasFiles
            ? `${fileCount} file${fileCount !== 1 ? 's' : ''} indexed — every answer, quiz and note is grounded in these.`
            : 'Add materials to power every tool — or import your course straight from Canvas below.'
        }
      />

      <Card padding="md" className="space-y-4">
        <h2 className="text-sm font-semibold text-zinc-200">Add materials</h2>
        <UploadZone courseId={courseId} prominent={!hasFiles} />
      </Card>

      <Card padding="md" className="space-y-4">
        <h2 className="text-sm font-semibold text-zinc-200">Indexed files</h2>
        <FileList courseId={courseId} />
      </Card>

      <IntegrationsPanel courseId={courseId} />
    </div>
  )
}
