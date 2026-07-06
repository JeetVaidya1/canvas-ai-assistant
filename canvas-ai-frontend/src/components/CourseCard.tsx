import { useNavigate } from 'react-router-dom'
import { GraduationCap, Trash2, FileText, ArrowRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { ProgressBar } from '@/components/ui/Progress'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useReadiness } from '@/hooks/useReadiness'
import { useUser } from '@/hooks/useUser'
import { usePrefetch } from '@/hooks/usePrefetch'
import { scoreTone } from '@/lib/score'

function timeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

interface CourseCardProps {
  courseId: string
  title: string
  lastVisit?: number
  onDelete: () => void
}

/**
 * Dashboard course card: identity + files + readiness at a glance.
 * Readiness renders only once cached/loaded — no spinner farm.
 */
export default function CourseCard({ courseId, title, lastVisit, onDelete }: CourseCardProps) {
  const navigate = useNavigate()
  const userId = useUser()
  const { prefetchCourse } = usePrefetch()
  const { data: files } = useCourseFiles(courseId)
  const fileCount = files?.length ?? 0
  const { data: readiness } = useReadiness(courseId, userId, { enabled: fileCount > 0 })

  const score = readiness ? Math.round(readiness.score_pct) : null
  const tone = score !== null ? scoreTone(score) : null

  return (
    <Card
      interactive
      accent
      onClick={() => navigate(`/course/${courseId}`)}
      onMouseEnter={() => prefetchCourse(courseId)}
      className="group h-full flex flex-col"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="w-10 h-10 rounded-xl bg-accent-wash border border-accent-line flex items-center justify-center">
          <GraduationCap className="w-5 h-5 text-accent" />
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          className="p-1.5 text-ink-faint hover:text-danger hover:bg-danger-wash rounded-lg transition-all opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
          aria-label={`Delete ${title}`}
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <h3 className="text-base font-semibold text-ink">{title}</h3>
      <div className="flex items-center gap-2.5 mt-2">
        <span className="inline-flex items-center gap-1.5 text-xs text-ink-soft">
          <FileText className="w-3.5 h-3.5" />
          {fileCount === 0 ? 'No files yet' : `${fileCount} file${fileCount !== 1 ? 's' : ''}`}
        </span>
        {lastVisit && (
          <>
            <span className="text-line-strong">·</span>
            <span className="text-xs text-ink-faint">{timeAgo(lastVisit)}</span>
          </>
        )}
      </div>

      <div className="mt-auto pt-4">
        {score !== null && tone ? (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] text-ink-faint">Exam readiness</span>
              <span className={`text-[11px] font-medium ${tone.text}`}>{score}% · {tone.label}</span>
            </div>
            <ProgressBar value={score} label={`${title} readiness`} />
          </div>
        ) : (
          <div className="flex items-center justify-between text-[11px] text-ink-faint">
            <span>{fileCount === 0 ? 'Add materials to get started' : 'Study to build your readiness score'}</span>
            <ArrowRight className="w-3.5 h-3.5 text-ink-faint group-hover:text-accent group-hover:translate-x-0.5 transition-all" />
          </div>
        )}
      </div>
    </Card>
  )
}
