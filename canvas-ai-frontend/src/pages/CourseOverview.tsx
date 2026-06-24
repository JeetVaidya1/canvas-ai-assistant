import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import IntegrationsPanel from '@/components/IntegrationsPanel'
import {
  Upload,
  FileText,
  X,
  MessageCircle,
  Target,
  BookOpen,
} from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles, useUploadFile, useDeleteFile } from '@/hooks/useCourseFiles'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
import CourseOverviewSkeleton from '@/components/skeletons/CourseOverviewSkeleton'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'

const QUICK_LINKS = [
  { label: 'Chat', path: '/chat', icon: MessageCircle },
  { label: 'Practice', path: '/practice', icon: Target },
  { label: 'Notes', path: '/notes', icon: BookOpen },
]

export default function CourseOverview() {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const { data: courses } = useCourses()
  const { data: files, isLoading: filesLoading } = useCourseFiles(courseId)
  const uploadFile = useUploadFile(courseId)
  const deleteFileM = useDeleteFile(courseId)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)

  const course = courses?.find((c) => c.course_id === courseId)

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) return
    setUploadProgress(10)
    const interval = setInterval(() => setUploadProgress((p) => Math.min(p + 10, 90)), 200)
    try {
      await uploadFile.mutateAsync(Array.from(selectedFiles))
      clearInterval(interval)
      setUploadProgress(100)
      setTimeout(() => {
        setUploadProgress(0)
        setSelectedFiles(null)
      }, 800)
    } catch {
      clearInterval(interval)
      setUploadProgress(0)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFiles(e.dataTransfer.files)
    }
  }

  if (!course) {
    return <CourseOverviewSkeleton />
  }

  const hasFiles = files && files.length > 0

  const pendingUploads = selectedFiles && (
    <div className="space-y-3 p-4 bg-zinc-900/60 rounded-lg border border-zinc-700/50">
      <div className="text-sm text-zinc-300">
        {Array.from(selectedFiles).map((f) => f.name).join(', ')}
      </div>
      <Button
        onClick={handleUpload}
        loading={uploadFile.isPending}
        className="w-full"
      >
        {uploadFile.isPending ? 'Uploading...' : 'Upload Files'}
      </Button>
      {uploadProgress > 0 && (
        <div className="w-full bg-zinc-700 rounded-full h-1">
          <div
            className="bg-gradient-brand h-1 rounded-full transition-all duration-300"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}
    </div>
  )

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <PageHeader
        eyebrow="Course"
        title={course.title}
        subtitle={hasFiles ? `${files.length} file${files.length !== 1 ? 's' : ''} uploaded` : undefined}
      />

      {/* Quick actions — only show when files exist */}
      {hasFiles && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {QUICK_LINKS.map((action) => (
            <Card
              key={action.label}
              interactive
              accent
              padding="sm"
              onClick={() => navigate(`/course/${courseId}${action.path}`)}
              className="flex items-center gap-3 group"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
                <action.icon className="w-5 h-5 text-cyan-300" />
              </div>
              <span className="text-sm font-medium text-zinc-200 group-hover:text-white transition-colors">{action.label}</span>
            </Card>
          ))}
        </div>
      )}

      {/* Export & integrations (Markdown / GitHub) */}
      {hasFiles && <IntegrationsPanel courseId={courseId || ''} />}

      {/* Files Section or Onboarding */}
      {!hasFiles && !filesLoading ? (
        // Onboarding state
        <Card accent padding="lg">
          <div className="w-12 h-12 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center mb-4">
            <Upload className="w-6 h-6 text-cyan-300" />
          </div>
          <h2 className="text-lg font-semibold text-zinc-100 mb-2">Upload your course materials</h2>
          <p className="text-sm text-zinc-500 mb-5 max-w-lg">
            Add your PDFs, slides, and documents. Vindexa will use them to generate quizzes, practice problems, study notes, and more.
          </p>

          {/* Drag-and-drop zone */}
          <label
            onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
            onDragLeave={(e) => { e.preventDefault(); setDragActive(false) }}
            onDrop={handleDrop}
            className={`flex flex-col items-center justify-center gap-2 px-6 py-8 rounded-xl border-2 border-dashed cursor-pointer transition-all ${
              dragActive
                ? 'border-cyan-500/60 bg-gradient-brand-soft'
                : 'border-zinc-700 bg-zinc-900/40 hover:border-cyan-500/40 hover:bg-zinc-900/60'
            }`}
          >
            <Upload className={`w-6 h-6 transition-colors ${dragActive ? 'text-cyan-300' : 'text-zinc-500'}`} />
            <span className="text-sm font-medium text-zinc-300">Drop files here or click to browse</span>
            <span className="text-xs text-zinc-600">PDF, DOCX, PPTX</span>
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.pptx"
              onChange={(e) => setSelectedFiles(e.target.files)}
              className="hidden"
            />
          </label>

          {/* Pending uploads */}
          {selectedFiles && <div className="mt-5">{pendingUploads}</div>}
        </Card>
      ) : (
        // Files list
        <Card padding="md" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-zinc-200">Files</h2>
            <label className="inline-flex items-center gap-2 font-medium rounded-lg transition-all text-white bg-gradient-brand hover:brightness-110 glow-brand-sm hover:glow-brand active:scale-[0.98] select-none cursor-pointer text-xs px-3 py-1.5">
              <Upload className="w-3.5 h-3.5" />
              Upload
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.pptx"
                onChange={(e) => setSelectedFiles(e.target.files)}
                className="hidden"
              />
            </label>
          </div>

          {/* Pending uploads */}
          {selectedFiles && pendingUploads}

          {/* File list */}
          {filesLoading ? (
            <LoadingSpinner size="sm" label="Loading files..." />
          ) : (
            <div className="space-y-1">
              {files?.map((filename) => (
                <div
                  key={filename}
                  className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-zinc-700/30 transition-colors group"
                >
                  <span className="text-zinc-300 text-sm truncate flex-1 flex items-center gap-2.5">
                    <FileText className="w-4 h-4 text-cyan-300/70 flex-shrink-0" />
                    {filename}
                  </span>
                  <button
                    onClick={() => deleteFileM.mutate(filename)}
                    className="p-1 text-zinc-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                    aria-label={`Delete ${filename}`}
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
