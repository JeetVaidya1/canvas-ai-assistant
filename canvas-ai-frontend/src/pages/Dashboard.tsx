import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Plus,
  Loader2,
  Trash2,
  X,
  ArrowRight,
  FileText,
  BookOpen,
} from 'lucide-react'
import { useCourses, useCreateCourse, useDeleteCourse } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import ConfirmDialog from '@/components/shared/ConfirmDialog'
import DashboardSkeleton from '@/components/skeletons/DashboardSkeleton'

function getGreeting() {
  const hour = new Date().getHours()
  if (hour < 12) return 'Good morning'
  if (hour < 18) return 'Good afternoon'
  return 'Good evening'
}

function timeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function CourseFileCount({ courseId }: { courseId: string }) {
  const { data: files } = useCourseFiles(courseId)
  if (!files || files.length === 0) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-zinc-500">
        <FileText className="w-3 h-3" />
        No files yet
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-zinc-400">
      <FileText className="w-3 h-3" />
      {files.length} file{files.length !== 1 ? 's' : ''}
    </span>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { data: courses, isLoading: coursesLoading } = useCourses()
  const createCourse = useCreateCourse()
  const deleteCourse = useDeleteCourse()
  const recentActivity = useRecentActivity()

  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [newCourseId, setNewCourseId] = useState('')
  const [newCourseTitle, setNewCourseTitle] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null)

  const handleCreate = async () => {
    if (!newCourseId.trim() || !newCourseTitle.trim()) return
    await createCourse.mutateAsync({ courseId: newCourseId.trim(), title: newCourseTitle.trim() })
    const id = newCourseId.trim()
    setNewCourseId('')
    setNewCourseTitle('')
    setShowCreateDialog(false)
    navigate(`/course/${id}`)
  }

  const handleDelete = async () => {
    if (!deleteTarget) return
    await deleteCourse.mutateAsync(deleteTarget.id)
    setDeleteTarget(null)
  }

  if (coursesLoading) {
    return <DashboardSkeleton />
  }

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  })

  // Build recent activity entries with course names
  const recentWithNames = recentActivity
    .map((entry) => {
      const course = courses?.find((c) => c.course_id === entry.courseId)
      if (!course) return null
      return { ...entry, courseTitle: course.title }
    })
    .filter(Boolean) as Array<{ courseId: string; page: string; timestamp: number; courseTitle: string }>

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
      {/* Greeting */}
      <div>
        <h1 className="text-2xl font-semibold text-zinc-50">{getGreeting()}</h1>
        <p className="text-sm text-zinc-500 mt-1">{today}</p>
      </div>

      {/* Continue studying */}
      {recentWithNames.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Continue studying</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {recentWithNames.map((entry) => (
              <button
                key={`${entry.courseId}-${entry.page}`}
                onClick={() => navigate(`/course/${entry.courseId}/${entry.page}`)}
                className="bg-zinc-800/80 border border-zinc-700/50 rounded-xl p-4 hover:bg-zinc-800 hover:border-zinc-600/50 cursor-pointer transition-all text-left flex items-center justify-between gap-3 group"
              >
                <div className="min-w-0">
                  <div className="text-sm font-medium text-zinc-200 truncate">{entry.courseTitle}</div>
                  <div className="text-xs text-zinc-500 capitalize mt-0.5">{entry.page} &middot; {timeAgo(entry.timestamp)}</div>
                </div>
                <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 flex-shrink-0 transition-colors" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Courses header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-medium text-zinc-200">Your Courses</h2>
        <button
          onClick={() => setShowCreateDialog(true)}
          className="bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Course
        </button>
      </div>

      {/* Course cards */}
      {!courses || courses.length === 0 ? (
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl py-16 px-8 text-center">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 flex items-center justify-center mx-auto mb-4">
            <BookOpen className="w-6 h-6 text-cyan-400" />
          </div>
          <h3 className="text-lg font-medium text-zinc-100 mb-2">Create your first course</h3>
          <p className="text-sm text-zinc-500 max-w-md mx-auto mb-6">
            Upload your course materials — PDFs, slides, docs — and Vindexa will turn them into quizzes, practice problems, notes, and more.
          </p>
          <button
            onClick={() => setShowCreateDialog(true)}
            className="bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            Get Started
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {courses.map((course) => {
            const lastVisit = recentActivity.find((e) => e.courseId === course.course_id)
            return (
              <div
                key={course.course_id}
                onClick={() => navigate(`/course/${course.course_id}`)}
                className="bg-zinc-800/80 border border-zinc-700/50 rounded-xl p-5 hover:bg-zinc-800 hover:border-zinc-600/50 cursor-pointer transition-all group relative"
              >
                <div className="absolute left-0 top-3 bottom-3 w-0.5 rounded-full bg-cyan-500" />
                <div className="flex items-start justify-between mb-3">
                  <h3 className="text-base font-medium text-zinc-100 pl-1">{course.title}</h3>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeleteTarget({ id: course.course_id, title: course.title })
                    }}
                    className="p-1.5 text-zinc-600 hover:text-red-400 hover:bg-zinc-700/50 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="flex items-center gap-3 pl-1">
                  <CourseFileCount courseId={course.course_id} />
                  {lastVisit && (
                    <>
                      <span className="text-zinc-700">&middot;</span>
                      <span className="text-xs text-zinc-600">{timeAgo(lastVisit.timestamp)}</span>
                    </>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Create Course Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => setShowCreateDialog(false)} />
          <div className="bg-zinc-900 border border-zinc-700/50 rounded-xl relative z-10 w-full max-w-md p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-medium text-zinc-100">Create New Course</h3>
              <button onClick={() => setShowCreateDialog(false)} className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded-lg transition-colors">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-400">Course ID</label>
                <input
                  type="text"
                  placeholder="e.g., CS101, MATH200"
                  value={newCourseId}
                  onChange={(e) => setNewCourseId(e.target.value)}
                  className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-600 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm transition-colors"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-400">Course Title</label>
                <input
                  type="text"
                  placeholder="e.g., Introduction to Computer Science"
                  value={newCourseTitle}
                  onChange={(e) => setNewCourseTitle(e.target.value)}
                  className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-600 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 outline-none text-sm transition-colors"
                />
              </div>
            </div>
            <div className="flex items-center justify-end gap-3 pt-1">
              <button
                onClick={() => setShowCreateDialog(false)}
                className="px-4 py-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={createCourse.isPending || !newCourseId.trim() || !newCourseTitle.trim()}
                className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                {createCourse.isPending ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Creating...</>
                ) : (
                  'Create Course'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Course"
        description={`Delete "${deleteTarget?.title}" and all its files? This cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
