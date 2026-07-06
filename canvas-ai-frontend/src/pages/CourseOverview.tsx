import { useParams } from 'react-router-dom'
import IntegrationsPanel from '@/components/IntegrationsPanel'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import CourseOverviewSkeleton from '@/components/skeletons/CourseOverviewSkeleton'
import UploadZone from '@/components/materials/UploadZone'
import FileList from '@/components/materials/FileList'
import CourseBrief from '@/components/materials/CourseBrief'
import { Card, PageHeader } from '@/components/ui/Card'

/**
 * Materials — the course's source of truth, in two columns on desktop:
 * LEFT the inputs (upload zone + indexed files), RIGHT the output — the
 * Course Brief showing what Vindexa understood from those files. On small
 * screens the Brief stacks under the files. Integrations sit below,
 * full-width: Canvas import matters most when the course is still empty.
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
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <PageHeader
        eyebrow="Materials"
        title={course.title}
        subtitle={
          hasFiles
            ? `${fileCount} file${fileCount !== 1 ? 's' : ''} indexed — every answer, quiz and note is grounded in these.`
            : 'Add materials to power every tool — or import your course straight from Canvas below.'
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <div className="space-y-6 min-w-0">
          <Card padding="md" className="space-y-4">
            <h2 className="text-sm font-semibold text-ink">Add materials</h2>
            <UploadZone courseId={courseId} prominent={!hasFiles} />
          </Card>

          <Card padding="md" className="space-y-4">
            <h2 className="text-sm font-semibold text-ink">Indexed files</h2>
            <FileList courseId={courseId} />
          </Card>
        </div>

        <div className="min-w-0">
          <CourseBrief courseId={courseId} hasFiles={hasFiles} />
        </div>
      </div>

      <IntegrationsPanel courseId={courseId} />
    </div>
  )
}
