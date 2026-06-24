import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  MessageCircle, Target, ClipboardList, Layers, BarChart3,
  Upload, ArrowRight, AlertTriangle,
} from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import { useUser } from '@/hooks/useUser'
import { getReadiness, type Readiness } from '@/lib/api'

const ACTIONS = [
  { key: 'learn', label: 'Learn', desc: 'Ask questions & get tutored, grounded in your files', icon: MessageCircle },
  { key: 'practice', label: 'Practice', desc: 'Quiz yourself or work adaptive problem sets', icon: Target },
  { key: 'exam', label: 'Exam', desc: 'Sit a timed mock exam with a readiness score', icon: ClipboardList },
  { key: 'kit', label: 'Study Kit', desc: 'Generate notes, flashcards & audio overviews', icon: Layers },
  { key: 'progress', label: 'Progress', desc: 'Track mastery, weak spots & your study plan', icon: BarChart3 },
] as const

function tone(score: number) {
  if (score >= 70) return { label: 'On track', ring: '#34d399', text: 'text-emerald-400' }
  if (score >= 40) return { label: 'Getting there', ring: '#fbbf24', text: 'text-amber-400' }
  return { label: 'At risk', ring: '#fb7185', text: 'text-rose-400' }
}

export default function CourseHome() {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const userId = useUser()
  const { data: courses } = useCourses()
  const { data: files } = useCourseFiles(courseId)
  const recent = useRecentActivity().filter((e) => e.courseId === courseId).slice(0, 4)
  const course = courses?.find((c) => c.course_id === courseId)
  const [readiness, setReadiness] = useState<Readiness | null>(null)

  useEffect(() => {
    if (!courseId) return
    let cancelled = false
    getReadiness(courseId, userId).then((r) => { if (!cancelled) setReadiness(r) }).catch(() => {})
    return () => { cancelled = true }
  }, [courseId, userId])

  const go = (path: string) => navigate(`/course/${courseId}${path ? `/${path}` : ''}`)
  const fileCount = files?.length ?? 0
  const score = readiness ? Math.round(readiness.score_pct) : null
  const t = score !== null ? tone(score) : null
  const circ = 2 * Math.PI * 32

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Course</p>
        <h1 className="text-3xl font-semibold text-zinc-50 tracking-tight">{course?.title ?? 'Course'}</h1>
        <p className="text-sm text-zinc-500 mt-1.5">
          {fileCount > 0
            ? `${fileCount} file${fileCount !== 1 ? 's' : ''} in your knowledge base`
            : 'No materials yet — upload your files to unlock the AI tools.'}
        </p>
      </div>

      {/* Readiness band */}
      {readiness && score !== null && t && (
        <Card accent padding="lg" className="flex flex-col md:flex-row md:items-center gap-5">
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="relative w-20 h-20">
              <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="32" fill="none" stroke="#27272a" strokeWidth="7" />
                <circle cx="40" cy="40" r="32" fill="none" stroke={t.ring} strokeWidth="7" strokeLinecap="round"
                  strokeDasharray={circ} strokeDashoffset={circ * (1 - score / 100)}
                  style={{ transition: 'stroke-dashoffset 0.8s ease' }} />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className={`text-xl font-bold ${t.text}`}>{score}%</span>
              </div>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1">Exam readiness</p>
              <p className={`text-lg font-semibold ${t.text}`}>{t.label}</p>
            </div>
          </div>
          <div className="flex-1 min-w-0">
            {readiness.gaps.length > 0 ? (
              <>
                <p className="text-xs text-zinc-500 mb-2 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> Focus on your weakest topics
                </p>
                <div className="flex flex-wrap gap-2">
                  {readiness.gaps.slice(0, 4).map((g) => (
                    <span key={g} className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-full px-2.5 py-0.5">{g}</span>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-zinc-400">No major gaps — keep practicing to hold your edge.</p>
            )}
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <Button variant="secondary" onClick={() => go('practice')}>Practice weak spots</Button>
            <Button onClick={() => go('exam')}>Mock exam</Button>
          </div>
        </Card>
      )}

      {/* Action grid */}
      <div>
        <h2 className="text-base font-semibold text-zinc-100 mb-3">What do you want to do?</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {ACTIONS.map((a) => (
            <Card key={a.key} interactive accent onClick={() => go(a.key)} className="group">
              <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center mb-4">
                <a.icon className="w-5 h-5 text-cyan-300" />
              </div>
              <h3 className="text-base font-semibold text-zinc-100 flex items-center gap-1.5">
                {a.label}
                <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-cyan-400 group-hover:translate-x-0.5 transition-all" />
              </h3>
              <p className="text-sm text-zinc-400 mt-1.5 leading-relaxed">{a.desc}</p>
            </Card>
          ))}
          <Card interactive onClick={() => go('materials')} className="group">
            <div className="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center mb-4">
              <Upload className="w-5 h-5 text-zinc-400" />
            </div>
            <h3 className="text-base font-semibold text-zinc-100 flex items-center gap-1.5">
              Materials
              <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-cyan-400 group-hover:translate-x-0.5 transition-all" />
            </h3>
            <p className="text-sm text-zinc-400 mt-1.5 leading-relaxed">
              {fileCount > 0 ? `${fileCount} uploaded — add or manage files` : 'Upload PDFs, slides & docs to get started'}
            </p>
          </Card>
        </div>
      </div>

      {/* Recent activity */}
      {recent.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold text-gradient-brand uppercase tracking-widest mb-3">Jump back in</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {recent.map((e) => (
              <Card key={`${e.page}`} interactive padding="sm" onClick={() => go(e.page)} className="flex items-center justify-between gap-3 group">
                <span className="text-sm text-zinc-300 capitalize">{e.page}</span>
                <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-cyan-400 transition-colors" />
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
