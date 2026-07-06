import { useNavigate, useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import {
  MessageCircle, Target, ClipboardList, Layers, BarChart3,
  Upload, ArrowRight, AlertTriangle, Sparkles, FileText, Clock,
} from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import { useUser } from '@/hooks/useUser'
import { useReadiness } from '@/hooks/useReadiness'
import { usePrefetch } from '@/hooks/usePrefetch'
import ErrorInline from '@/components/shared/ErrorInline'

// One treatment for every action (Paper & Ink): paper-deep icon well with an
// ink icon — the pen-blue accent appears only on the hover arrow.
const ACTIONS: ReadonlyArray<{
  key: string; label: string; desc: string; capability: string; icon: typeof MessageCircle
}> = [
  { key: 'learn', label: 'Learn', desc: 'Chat, Socratic tutor & Feynman checks — grounded in your files.', capability: 'Cites exact pages', icon: MessageCircle },
  { key: 'practice', label: 'Practice', desc: 'Rapid quiz drills or deep adaptive problem sets.', capability: 'Adapts to mastery', icon: Target },
  { key: 'exam', label: 'Exam', desc: 'Sit a timed mock exam, graded with concept breakdown.', capability: 'Timed & graded', icon: ClipboardList },
  { key: 'kit', label: 'Study Kit', desc: 'Generate grounded notes with spaced-repetition flashcards.', capability: 'Notes · flashcards', icon: Layers },
  { key: 'progress', label: 'Progress', desc: 'Mastery map, weak spots & an AI study plan.', capability: 'Concept graph', icon: BarChart3 },
]

export default function CourseHome() {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const userId = useUser()
  const { data: courses } = useCourses()
  const { data: files, isLoading: filesLoading } = useCourseFiles(courseId)
  const recent = useRecentActivity().filter((e) => e.courseId === courseId).slice(0, 4)
  const course = courses?.find((c) => c.course_id === courseId)
  const readinessQuery = useReadiness(courseId, userId)
  const readiness = readinessQuery.data ?? null
  const readinessLoading = readinessQuery.isPending
  const { prefetchLearn, prefetchPractice, prefetchStudyKit, prefetchProgress } = usePrefetch()

  const go = (path: string) => navigate(`/course/${courseId}${path ? `/${path}` : ''}`)

  // Warm the destination's primary data while the cursor hovers its card.
  const prefetchFor = (key: string) => {
    if (!courseId) return
    if (key === 'learn') prefetchLearn()
    else if (key === 'practice') prefetchPractice(courseId)
    else if (key === 'kit') prefetchStudyKit(courseId)
    else if (key === 'progress') prefetchProgress(courseId)
  }
  const fileCount = files?.length ?? 0
  const hasFiles = fileCount > 0
  const score = readiness ? Math.round(readiness.score_pct) : null
  const t = score !== null ? scoreTone(score) : null

  const lastStudied = recent[0]?.page

  return (
    <div className="max-w-5xl mx-auto px-6 py-9 space-y-9">
      {/* ── Header ───────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-5"
      >
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-ink-faint mb-2">Course</p>
          <h1 className="font-display text-[2rem] leading-tight font-semibold text-ink tracking-tight truncate">
            {course?.title ?? 'Course'}
          </h1>
          <div className="flex items-center gap-3 mt-2.5 text-sm text-ink-soft">
            <span className="inline-flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-ink-faint" />
              {filesLoading ? (
                <span className="inline-block h-3.5 w-24 rounded bg-paper-deep animate-pulse align-middle" />
              ) : (
                <span>{hasFiles ? `${fileCount} file${fileCount !== 1 ? 's' : ''} indexed` : 'No materials yet'}</span>
              )}
            </span>
            {lastStudied && (
              <span className="inline-flex items-center gap-1.5 text-ink-faint">
                <Clock className="w-4 h-4" /> Last: <span className="capitalize text-ink-soft">{lastStudied}</span>
              </span>
            )}
          </div>
        </div>
        <Button
          size="lg" leftIcon={<Sparkles className="w-4 h-4" />}
          onClick={() => go(lastStudied ?? 'learn')}
          className="flex-shrink-0"
        >
          {lastStudied ? 'Continue studying' : 'Ask your course'}
        </Button>
      </motion.div>

      {/* ── Readiness hero ───────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
      >
        {readinessQuery.isError ? (
          <Card padding="lg" elevation={2}>
            <ErrorInline
              message="Couldn't load your readiness score."
              onRetry={() => void readinessQuery.refetch()}
            />
          </Card>
        ) : readinessLoading && !readiness ? (
          <Card padding="lg" elevation={2} className="flex items-center gap-5">
            <div className="w-[104px] h-[104px] rounded-full bg-paper-deep animate-pulse flex-shrink-0" />
            <div className="flex-1 space-y-3">
              <div className="h-3 w-32 rounded bg-paper-deep animate-pulse" />
              <div className="h-5 w-40 rounded bg-paper-deep animate-pulse" />
              <div className="h-3 w-3/4 rounded bg-paper-deep animate-pulse" />
            </div>
          </Card>
        ) : readiness && score !== null && t ? (
          <Card accent padding="lg" elevation={2} className="relative overflow-hidden">
            <div className="flex flex-col md:flex-row md:items-center gap-6">
              {/* Ring */}
              <div className="flex items-center gap-5 flex-shrink-0">
                <ProgressRing value={score}>
                  <span className={`text-2xl font-bold tnum ${t.text}`}>{score}%</span>
                  <span className="text-[10px] uppercase tracking-widest text-ink-faint mt-0.5">ready</span>
                </ProgressRing>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-ink-faint mb-1">Exam readiness</p>
                  <p className={`text-xl font-semibold ${t.text}`}>{t.label}</p>
                  <p className="text-xs text-ink-faint mt-1 max-w-[16rem]">Estimated from your topic mastery.</p>
                </div>
              </div>

              {/* Gaps */}
              <div className="flex-1 min-w-0 md:border-l md:border-line md:pl-6">
                {readiness.gaps.length > 0 ? (
                  <>
                    <p className="text-xs text-ink-soft mb-2.5 flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5 text-warning" /> Weakest topics to focus on
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {readiness.gaps.slice(0, 5).map((g) => (
                        <button key={g} onClick={() => go('practice')} className="focus-ring rounded-full">
                          <Badge tone="warning" className="cursor-pointer hover:bg-warning/15 transition-colors">{g}</Badge>
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-ink-soft">No major gaps — keep practicing to hold your edge.</p>
                )}
                <div className="flex gap-2.5 mt-4">
                  <Button variant="secondary" size="sm" onClick={() => go('practice')}>Practice weak spots</Button>
                  <Button size="sm" onClick={() => go('exam')}>Take mock exam</Button>
                </div>
              </div>
            </div>
          </Card>
        ) : (
          /* No readiness yet (e.g. no activity) — invite first action */
          <Card padding="lg" elevation={2} className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="text-base font-semibold text-ink">Build your readiness score</p>
              <p className="text-sm text-ink-soft mt-1">
                {hasFiles ? 'Ask questions or take a quiz and we’ll start tracking your mastery.' : 'Upload your course materials to unlock the AI tools.'}
              </p>
            </div>
            <Button onClick={() => go(hasFiles ? 'learn' : 'materials')} className="flex-shrink-0">
              {hasFiles ? 'Start learning' : 'Upload materials'}
            </Button>
          </Card>
        )}
      </motion.div>

      {/* ── Action grid ──────────────────────────────────── */}
      <div>
        <div className="section-head mb-3.5">
          <span className="section-num">01</span>
          <h2 className="text-sm font-semibold text-ink">What do you want to do?</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {ACTIONS.map((a, i) => (
            <motion.div
              key={a.key}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
            >
              <Card
                interactive
                accent
                onClick={() => go(a.key)}
                onMouseEnter={() => prefetchFor(a.key)}
                padding="lg"
                className="group h-full"
              >
                <div className="flex items-start justify-between">
                  <div className="w-11 h-11 rounded-xl bg-paper-deep border border-line flex items-center justify-center mb-4">
                    <a.icon className="w-5 h-5 text-ink-soft" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-ink-faint group-hover:text-accent group-hover:translate-x-0.5 transition-all mt-1" />
                </div>
                <h3 className="text-base font-semibold text-ink">{a.label}</h3>
                <p className="text-sm text-ink-soft mt-1.5 leading-relaxed">{a.desc}</p>
                <Badge tone="neutral" className="mt-3">{a.capability}</Badge>
              </Card>
            </motion.div>
          ))}

          {/* Materials — the source-of-truth tile */}
          <motion.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 + ACTIONS.length * 0.04, ease: [0.22, 1, 0.36, 1] }}
          >
            <Card interactive onClick={() => go('materials')} padding="lg" className="group h-full border-dashed">
              <div className="flex items-start justify-between">
                <div className="w-11 h-11 rounded-xl bg-paper-deep border border-line flex items-center justify-center mb-4">
                  <Upload className="w-5 h-5 text-ink-soft" />
                </div>
                <ArrowRight className="w-4 h-4 text-ink-faint group-hover:text-accent group-hover:translate-x-0.5 transition-all mt-1" />
              </div>
              <h3 className="text-base font-semibold text-ink">Materials</h3>
              <p className="text-sm text-ink-soft mt-1.5 leading-relaxed">
                {hasFiles ? `${fileCount} file${fileCount !== 1 ? 's' : ''} indexed — add or manage your knowledge base.` : 'Upload PDFs, slides & docs to power every tool.'}
              </p>
            </Card>
          </motion.div>
        </div>
      </div>

      {/* ── Jump back in ─────────────────────────────────── */}
      {recent.length > 0 && (
        <div>
          <div className="section-head mb-3">
            <span className="section-num">02</span>
            <h2 className="text-sm font-semibold text-ink">Jump back in</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {recent.map((e) => (
              <Card key={e.page} interactive padding="sm" onClick={() => go(e.page)} className="flex items-center justify-between gap-3 group">
                <span className="text-sm text-ink capitalize">{e.page}</span>
                <ArrowRight className="w-4 h-4 text-ink-faint group-hover:text-accent group-hover:translate-x-0.5 transition-all" />
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
