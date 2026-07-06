import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import {
  Plus, ArrowRight, BookOpen, MessageCircle, Target, ClipboardList, Layers, BarChart3,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { useCourses, useDeleteCourse } from '@/hooks/useCourses'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import ConfirmDialog from '@/components/shared/ConfirmDialog'
import DashboardSkeleton from '@/components/skeletons/DashboardSkeleton'
import JoinClassPanel from '@/components/JoinClassPanel'
import CreateCourseModal from '@/components/CreateCourseModal'
import CourseCard from '@/components/CourseCard'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'

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
  return `${Math.floor(hours / 24)}d ago`
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
const pageMeta = (page: string) =>
  PAGE_META[page] ?? { icon: BookOpen, tint: 'text-zinc-300 bg-white/[0.05] border-white/10', label: page }

export default function Dashboard() {
  const navigate = useNavigate()
  const { data: courses, isLoading: coursesLoading } = useCourses()
  const deleteCourse = useDeleteCourse()
  const recentActivity = useRecentActivity()

  const [showCreate, setShowCreate] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; title: string } | null>(null)

  const handleDelete = async () => {
    if (!deleteTarget) return
    await deleteCourse.mutateAsync(deleteTarget.id)
    setDeleteTarget(null)
  }

  if (coursesLoading) return <DashboardSkeleton />

  const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })
  const hasCourses = !!courses && courses.length > 0

  const recentWithNames = recentActivity
    .map((entry) => {
      const course = courses?.find((c) => c.course_id === entry.courseId)
      if (!course) return null
      return { ...entry, courseTitle: course.title }
    })
    .filter(Boolean) as Array<{ courseId: string; page: string; timestamp: number; courseTitle: string }>

  return (
    <div className="max-w-5xl mx-auto px-6 py-9 space-y-9">
      {/* Header: greeting + primary action */}
      <motion.div
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
        className="flex items-end justify-between gap-4"
      >
        <div>
          <h1 className="text-[2rem] leading-tight font-semibold tracking-tight text-zinc-50">{getGreeting()}</h1>
          <p className="text-sm text-zinc-400 mt-1.5">{today}</p>
        </div>
        {hasCourses && (
          <Button onClick={() => setShowCreate(true)} leftIcon={<Plus className="w-4 h-4" />} className="flex-shrink-0">
            New course
          </Button>
        )}
      </motion.div>

      {/* Continue studying */}
      {recentWithNames.length > 0 && (
        <div className="space-y-3.5">
          <h2 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Continue studying</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {recentWithNames.map((entry) => {
              const m = pageMeta(entry.page)
              return (
                <Card
                  key={`${entry.courseId}-${entry.page}`}
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
              )
            })}
          </div>
        </div>
      )}

      {/* Courses */}
      {!hasCourses ? (
        <Card padding="none" elevation={2}>
          <EmptyState
            icon={<BrandMark className="h-10 w-10" />}
            title="Create your first course"
            description="Add your course materials — PDFs, slides, docs, or a Canvas import — and Vindexa turns them into grounded chat, practice, exams, and notes."
            action={<Button size="lg" onClick={() => setShowCreate(true)}>Get started</Button>}
            className="py-16"
          />
        </Card>
      ) : (
        <div className="space-y-3.5">
          <h2 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-[0.18em]">Your courses</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {courses.map((course) => (
              <CourseCard
                key={course.course_id}
                courseId={course.course_id}
                title={course.title}
                lastVisit={recentActivity.find((e) => e.courseId === course.course_id)?.timestamp}
                onDelete={() => setDeleteTarget({ id: course.course_id, title: course.title })}
              />
            ))}
          </div>
        </div>
      )}

      {/* Shared classes — secondary, lives below your own courses */}
      <JoinClassPanel />

      <CreateCourseModal open={showCreate} onClose={() => setShowCreate(false)} />

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
