import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import {
  Plus, Trash2, X, ArrowRight, FileText, BookOpen, GraduationCap,
  MessageCircle, Target, ClipboardList, Layers, BarChart3,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { useCourses, useCreateCourse, useDeleteCourse } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import ConfirmDialog from '@/components/shared/ConfirmDialog'
import DashboardSkeleton from '@/components/skeletons/DashboardSkeleton'
import JoinClassPanel from '@/components/JoinClassPanel'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'

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

// Icon + tint per destination, so "continue studying" cards read at a glance.
const PAGE_META: Record<string, { icon: typeof MessageCircle; tint: string; label: string }> = {
  learn: { icon: MessageCircle, tint: 'text-cyan-300 bg-cyan-500/12 border-cyan-400/20', label: 'Learn' },
  chat: { icon: MessageCircle, tint: 'text-cyan-300 bg-cyan-500/12 border-cyan-400/20', label: 'Chat' },
  practice: { icon: Target, tint: 'text-sky-300 bg-blue-500/12 border-blue-400/20', label: 'Practice' },
  quiz: { icon: Target, tint: 'text-sky-300 bg-blue-500/12 border-blue-400/20', label: 'Quiz' },
  exam: { icon: ClipboardList, tint: 'text-rose-300 bg-rose-500/12 border-rose-400/20', label: 'Exam' },
  exams: { icon: ClipboardList, tint: 'text-rose-300 bg-rose-500/12 border-rose-400/20', label: 'Exam' },
  kit: { icon: Layers, tint: 'text-emerald-300 bg-emerald-500/12 border-emerald-400/20', label: 'Study Kit' },
  notes: { icon: Layers, tint: 'text-emerald-300 bg-emerald-500/12 border-emerald-400/20', label: 'Notes' },
  progress: { icon: BarChart3, tint: 'text-sky-300 bg-sky-500/12 border-sky-400/20', label: 'Progress' },
}
const pageMeta = (page: string) => PAGE_META[page] ?? { icon: BookOpen, tint: 'text-zinc-300 bg-white/[0.05] border-white/10', label: page }

function CourseFileCount({ courseId }: { courseId: string }) {
  const { data: files } = useCourseFiles(courseId)
  const count = files?.length ?? 0
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-zinc-400">
      <FileText className="w-3.5 h-3.5" />
      {count === 0 ? 'No files yet' : `${count} file${count !== 1 ? 's' : ''}`}
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

  if (coursesLoading) return <DashboardSkeleton />

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  })

  const recentWithNames = recentActivity
    .map((entry) => {
      const course = courses?.find((c) => c.course_id === entry.courseId)
      if (!course) return null
      return { ...entry, courseTitle: course.title }
    })
    .filter(Boolean) as Array<{ courseId: string; page: string; timestamp: number; courseTitle: string }>

  const inputClass =
    'w-full px-3.5 py-2.5 bg-white/[0.03] border border-white/10 rounded-lg text-zinc-100 placeholder-zinc-600 ' +
    'focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25 outline-none text-sm transition-colors'

  return (
    <div className="max-w-5xl mx-auto px-6 py-9 space-y-9">
      {/* Greeting */}
      <motion.div
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <h1 className="text-[2rem] leading-tight font-semibold tracking-tight text-zinc-50">{getGreeting()}</h1>
        <p className="text-sm text-zinc-400 mt-1.5">{today}</p>
      </motion.div>

      {/* Continue studying */}
      {recentWithNames.length > 0 && (
        <div className="space-y-3.5">
          <h2 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Continue studying</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {recentWithNames.map((entry, i) => {
              const m = pageMeta(entry.page)
              return (
                <motion.div
                  key={`${entry.courseId}-${entry.page}`}
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.04 * i, ease: [0.22, 1, 0.36, 1] }}
                >
                  <Card
                    interactive padding="sm"
                    onClick={() => navigate(`/course/${entry.courseId}/${entry.page}`)}
                    className="flex items-center gap-3 group h-full"
                  >
                    <div className={`w-9 h-9 rounded-lg border flex items-center justify-center flex-shrink-0 ${m.tint}`}>
                      <m.icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-zinc-100 truncate">{entry.courseTitle}</div>
                      <div className="text-xs text-zinc-500 mt-0.5">{m.label} · {timeAgo(entry.timestamp)}</div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-cyan-300 group-hover:translate-x-0.5 flex-shrink-0 transition-all" />
                  </Card>
                </motion.div>
              )
            })}
          </div>
        </div>
      )}

      {/* Join a shared class */}
      <JoinClassPanel />

      {/* Courses header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-zinc-100">Your courses</h2>
        <Button onClick={() => setShowCreateDialog(true)} leftIcon={<Plus className="w-4 h-4" />}>
          New course
        </Button>
      </div>

      {/* Course cards */}
      {!courses || courses.length === 0 ? (
        <Card padding="none" elevation={2} className="py-16 px-8 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">Create your first course</h3>
          <p className="text-sm text-zinc-400 max-w-md mx-auto mb-6">
            Upload your course materials — PDFs, slides, docs — and Vindexa turns them into quizzes, practice problems, notes, and more.
          </p>
          <Button size="lg" onClick={() => setShowCreateDialog(true)}>Get started</Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {courses.map((course, i) => {
            const lastVisit = recentActivity.find((e) => e.courseId === course.course_id)
            return (
              <motion.div
                key={course.course_id}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.05 + 0.04 * i, ease: [0.22, 1, 0.36, 1] }}
              >
                <Card
                  interactive accent
                  onClick={() => navigate(`/course/${course.course_id}`)}
                  className="group h-full"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center">
                      <GraduationCap className="w-5 h-5 text-cyan-300" />
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setDeleteTarget({ id: course.course_id, title: course.title })
                      }}
                      className="p-1.5 text-zinc-600 hover:text-rose-400 hover:bg-white/[0.06] rounded-lg transition-all opacity-0 group-hover:opacity-100"
                      aria-label="Delete course"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <h3 className="text-base font-semibold text-zinc-50">{course.title}</h3>
                  <div className="flex items-center gap-2.5 mt-2">
                    <CourseFileCount courseId={course.course_id} />
                    {lastVisit && (
                      <>
                        <span className="text-zinc-700">·</span>
                        <span className="text-xs text-zinc-500">{timeAgo(lastVisit.timestamp)}</span>
                      </>
                    )}
                  </div>
                </Card>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Create Course Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => setShowCreateDialog(false)} />
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }} animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="relative z-10 w-full max-w-md"
          >
            <Card padding="lg" elevation={3} className="space-y-5 !bg-[#16161b]">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-zinc-100">Create new course</h3>
                <button onClick={() => setShowCreateDialog(false)} className="p-1.5 text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.06] rounded-lg transition-colors" aria-label="Close">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">Course ID</label>
                  <input type="text" placeholder="e.g., CS101, MATH200" value={newCourseId} onChange={(e) => setNewCourseId(e.target.value)} className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">Course title</label>
                  <input type="text" placeholder="e.g., Introduction to Computer Science" value={newCourseTitle} onChange={(e) => setNewCourseTitle(e.target.value)} className={inputClass} />
                </div>
              </div>
              <div className="flex items-center justify-end gap-2 pt-1">
                <Button variant="ghost" onClick={() => setShowCreateDialog(false)}>Cancel</Button>
                <Button
                  onClick={handleCreate}
                  loading={createCourse.isPending}
                  disabled={!newCourseId.trim() || !newCourseTitle.trim()}
                >
                  Create course
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete course"
        description={`Delete "${deleteTarget?.title}" and all its files? This cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
