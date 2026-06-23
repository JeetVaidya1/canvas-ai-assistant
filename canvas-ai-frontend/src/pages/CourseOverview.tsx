import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Upload,
  FileText,
  X,
  Loader2,
  MessageCircle,
  Target,
  BookOpen,
} from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles, useUploadFile, useDeleteFile } from '@/hooks/useCourseFiles'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
import CourseOverviewSkeleton from '@/components/skeletons/CourseOverviewSkeleton'

export default function CourseOverview() {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const { data: courses } = useCourses()
  const { data: files, isLoading: filesLoading } = useCourseFiles(courseId)
  const uploadFile = useUploadFile(courseId)
  const deleteFileM = useDeleteFile(courseId)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)

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

  if (!course) {
    return <CourseOverviewSkeleton />
  }

  const hasFiles = files && files.length > 0

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-zinc-50">{course.title}</h1>
        {hasFiles && (
          <p className="text-sm text-zinc-500 mt-1">{files.length} file{files.length !== 1 ? 's' : ''} uploaded</p>
        )}
      </div>

      {/* Quick actions — only show when files exist */}
      {hasFiles && (
        <div className="flex items-center gap-2">
          {[
            { label: 'Chat', path: '/chat', icon: MessageCircle },
            { label: 'Practice', path: '/practice', icon: Target },
            { label: 'Notes', path: '/notes', icon: BookOpen },
          ].map((action) => (
            <button
              key={action.label}
              onClick={() => navigate(`/course/${courseId}${action.path}`)}
              className="flex items-center gap-1.5 px-3.5 py-2 bg-zinc-800/80 border border-zinc-700/50 rounded-lg text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 hover:border-zinc-600/50 transition-all"
            >
              <action.icon className="w-3.5 h-3.5" />
              {action.label}
            </button>
          ))}
        </div>
      )}

      {/* Files Section or Onboarding */}
      {!hasFiles && !filesLoading ? (
        // Onboarding state
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-8">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mb-4">
            <Upload className="w-6 h-6 text-cyan-400" />
          </div>
          <h2 className="text-lg font-medium text-zinc-100 mb-2">Upload your course materials</h2>
          <p className="text-sm text-zinc-500 mb-5 max-w-lg">
            Add your PDFs, slides, and documents. Vindexa will use them to generate quizzes, practice problems, study notes, and more.
          </p>
          <label className="inline-flex items-center gap-2 px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg cursor-pointer transition-colors text-sm font-medium">
            <Upload className="w-4 h-4" />
            Upload Files
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.pptx"
              onChange={(e) => setSelectedFiles(e.target.files)}
              className="hidden"
            />
          </label>

          {/* Pending uploads */}
          {selectedFiles && (
            <div className="space-y-3 p-4 bg-zinc-900/60 rounded-lg border border-zinc-700/50 mt-5">
              <div className="text-sm text-zinc-300">
                {Array.from(selectedFiles).map((f) => f.name).join(', ')}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploadFile.isPending}
                className="w-full bg-emerald-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-emerald-500 disabled:opacity-50 transition-colors"
              >
                {uploadFile.isPending ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" /> Uploading...
                  </span>
                ) : (
                  'Upload Files'
                )}
              </button>
              {uploadProgress > 0 && (
                <div className="w-full bg-zinc-700 rounded-full h-1">
                  <div
                    className="bg-emerald-500 h-1 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        // Files list
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-zinc-200">Files</h2>
            <label className="flex items-center gap-2 px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg cursor-pointer transition-colors text-xs font-medium">
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
          {selectedFiles && (
            <div className="space-y-3 p-3 bg-zinc-900/60 rounded-lg border border-zinc-700/50">
              <div className="text-sm text-zinc-300">
                {Array.from(selectedFiles).map((f) => f.name).join(', ')}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploadFile.isPending}
                className="w-full bg-emerald-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-emerald-500 disabled:opacity-50 transition-colors"
              >
                {uploadFile.isPending ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" /> Uploading...
                  </span>
                ) : (
                  'Upload Files'
                )}
              </button>
              {uploadProgress > 0 && (
                <div className="w-full bg-zinc-700 rounded-full h-1">
                  <div
                    className="bg-emerald-500 h-1 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              )}
            </div>
          )}

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
                    <FileText className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                    {filename}
                  </span>
                  <button
                    onClick={() => deleteFileM.mutate(filename)}
                    className="p-1 text-zinc-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
