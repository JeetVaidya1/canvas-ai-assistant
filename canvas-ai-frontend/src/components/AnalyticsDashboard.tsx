// src/components/AnalyticsDashboard.tsx
import { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts'
import {
  TrendingUp,
  Target,
  Brain,
  Award,
  AlertCircle,
  BookOpen,
  Zap,
  BarChart3,
  Calendar,
  Flame
} from 'lucide-react'
import { getReadiness, getConceptGraph, type Readiness, type ConceptBlocker } from '@/lib/api'
import { Card, PageHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ReviewPanel from './ReviewPanel'

interface AnalyticsData {
  topics_progress: Array<{
    topic: string
    mastery_level: number
    review_count: number
    last_reviewed: string
  }>
  study_streak: number
  weak_areas: string[]
  study_recommendations: string[]
  total_questions: number
  avg_confidence: number
  study_time_trend: Array<{
    date: string
    questions: number
    duration_minutes?: number
    avg_confidence?: number
  }>
}

const MASTERY_COLORS = ['#ef4444', '#f59e0b', '#06b6d4', '#10b981']

function masteryColor(level: number): string {
  if (level >= 0.8) return MASTERY_COLORS[3]
  if (level >= 0.7) return MASTERY_COLORS[2]
  if (level >= 0.5) return MASTERY_COLORS[1]
  return MASTERY_COLORS[0]
}

function shortDate(iso: string): string {
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

interface AnalyticsDashboardProps {
  courseId: string
  userId: string
}

function readinessTone(score: number): { ring: string; text: string; label: string } {
  if (score >= 80) return { ring: '#10b981', text: 'text-emerald-400', label: 'On track' }
  if (score >= 60) return { ring: '#06b6d4', text: 'text-cyan-400', label: 'Getting there' }
  if (score >= 40) return { ring: '#f59e0b', text: 'text-amber-400', label: 'Needs work' }
  return { ring: '#ef4444', text: 'text-red-400', label: 'At risk' }
}

function ReadinessHero({ readiness }: { readiness: Readiness }) {
  const score = Math.round(readiness.score_pct)
  const tone = readinessTone(score)
  const circumference = 2 * Math.PI * 52
  const offset = circumference * (1 - score / 100)
  return (
    <Card accent padding="lg" className="flex flex-col md:flex-row items-center gap-6">
      <div className="relative w-32 h-32 flex-shrink-0">
        <svg className="w-32 h-32 -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" fill="none" stroke="#27272a" strokeWidth="10" />
          <circle
            cx="60" cy="60" r="52" fill="none" stroke={tone.ring} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${tone.text}`}>{score}%</span>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest">ready</span>
        </div>
      </div>
      <div className="flex-1 text-center md:text-left">
        <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Exam readiness</p>
        <div className="flex items-center justify-center md:justify-start gap-2 mb-1.5">
          <h2 className="text-xl font-semibold text-zinc-50 tracking-tight">{tone.label}</h2>
        </div>
        <p className="text-sm text-zinc-400 mb-3">
          {readiness.has_past_papers
            ? 'Weighted by how often each topic shows up on your past papers.'
            : 'Based on your topic mastery. Upload a past paper to weight by what’s actually tested.'}
          {readiness.confidence === 'low' && ' Study more to sharpen this estimate.'}
        </p>
        {readiness.gaps.length > 0 ? (
          <div className="flex flex-wrap gap-2 justify-center md:justify-start">
            <span className="text-xs text-zinc-500 self-center">Biggest gaps:</span>
            {readiness.gaps.map((g) => (
              <span key={g} className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-full px-2.5 py-0.5">
                {g}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-xs text-emerald-400">No major gaps — keep reviewing to hold your edge.</p>
        )}
      </div>
    </Card>
  )
}

export default function AnalyticsDashboard({ courseId, userId }: AnalyticsDashboardProps) {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [blockers, setBlockers] = useState<ConceptBlocker[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadAnalytics()
  }, [courseId, userId])

  // Concept-graph blockers load separately (first call builds the graph; slow).
  useEffect(() => {
    if (!courseId) return
    let cancelled = false
    getConceptGraph(courseId, userId)
      .then((g) => { if (!cancelled) setBlockers(g.blockers || []) })
      .catch(() => { /* non-blocking */ })
    return () => { cancelled = true }
  }, [courseId, userId])

  const loadAnalytics = async () => {
    try {
      const [response, r] = await Promise.all([
        fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/analytics/${courseId}/${userId}`),
        getReadiness(courseId, userId).catch(() => null),
      ])
      if (response.ok) {
        const data = await response.json()
        setAnalytics(data.analytics)
      }
      setReadiness(r)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-zinc-800 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-zinc-800 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <Card padding="none" className="py-16 px-8 text-center">
          <div className="w-14 h-14 rounded-2xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center mx-auto mb-5">
            <BarChart3 className="w-7 h-7 text-cyan-300" />
          </div>
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">No analytics data yet</h3>
          <p className="text-sm text-zinc-500">Start studying to see your progress!</p>
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Header */}
      <PageHeader
        eyebrow="Analytics"
        title="Learning Analytics"
        subtitle="Track your progress and identify areas for improvement"
        actions={
          <Button variant="secondary" onClick={loadAnalytics} leftIcon={<TrendingUp className="w-4 h-4" />}>
            Refresh
          </Button>
        }
      />

      {/* Exam readiness hero */}
      {readiness && <ReadinessHero readiness={readiness} />}

      {/* Mistake-driven review queue (hidden when nothing is due) */}
      <ReviewPanel courseId={courseId} userId={userId} />

      {/* Prerequisite gaps — fix the foundation first */}
      {blockers.length > 0 && (
        <Card padding="md">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center flex-shrink-0">
              <Brain className="w-4 h-4 text-violet-400" />
            </div>
            <h2 className="text-sm font-semibold text-zinc-100">Fix the foundation first</h2>
          </div>
          <p className="text-xs text-zinc-500 mb-4 ml-[42px]">You're weak on these, and so are their prerequisites — start upstream.</p>
          <div className="space-y-2 ml-[42px]">
            {blockers.slice(0, 5).map((b, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded px-2 py-0.5 text-xs">
                  {b.prerequisite} <span className="text-zinc-500">({Math.round(b.prerequisite_pct)}%)</span>
                </span>
                <span className="text-zinc-600">→ unlocks →</span>
                <span className="text-zinc-300 text-xs">{b.concept} <span className="text-zinc-500">({Math.round(b.concept_pct)}%)</span></span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Study Streak</p>
              <p className="text-3xl font-bold text-gradient-brand mt-1.5 leading-none">{analytics.study_streak}</p>
              <p className="text-xs text-zinc-500 mt-1.5">days</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
              <Flame className="w-5 h-5 text-cyan-300" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Questions Asked</p>
              <p className="text-3xl font-bold text-gradient-brand mt-1.5 leading-none">{analytics.total_questions}</p>
              <p className="text-xs text-zinc-500 mt-1.5">total</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-cyan-300" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Avg Confidence</p>
              <p className="text-3xl font-bold text-gradient-brand mt-1.5 leading-none">{Math.round(analytics.avg_confidence * 100)}%</p>
              <p className="text-xs text-zinc-500 mt-1.5">score</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
              <Target className="w-5 h-5 text-cyan-300" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Topics Studied</p>
              <p className="text-3xl font-bold text-gradient-brand mt-1.5 leading-none">{analytics.topics_progress.length}</p>
              <p className="text-xs text-zinc-500 mt-1.5">concepts</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
              <Brain className="w-5 h-5 text-cyan-300" />
            </div>
          </div>
        </Card>
      </div>

      {/* Topic Progress */}
      <Card padding="lg">
        <div className="flex items-center gap-2.5 mb-6">
          <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-lg font-semibold text-zinc-50 tracking-tight">Topic Mastery Progress</h2>
        </div>

        {analytics.topics_progress.length === 0 ? (
          <div className="text-center py-8 text-zinc-500">
            <Brain className="w-12 h-12 text-zinc-600 mx-auto mb-3" />
            <p className="text-sm">No topics studied yet. Start asking questions to see progress!</p>
          </div>
        ) : (
          <div style={{ height: Math.max(160, analytics.topics_progress.length * 40) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={analytics.topics_progress.map((t) => ({
                  topic: t.topic,
                  mastery: Math.round(t.mastery_level * 100),
                }))}
                layout="vertical"
                margin={{ left: 8, right: 24, top: 4, bottom: 4 }}
              >
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" unit="%" />
                <YAxis type="category" dataKey="topic" width={150} tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" />
                <Tooltip
                  cursor={{ fill: '#27272a' }}
                  contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                  formatter={(value) => [`${value}%`, 'Mastery']}
                />
                <Bar dataKey="mastery" radius={[0, 4, 4, 0]}>
                  {analytics.topics_progress.map((t, i) => (
                    <Cell key={i} fill={masteryColor(t.mastery_level)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* Study Activity Trend */}
      {analytics.study_time_trend.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card padding="lg">
            <div className="flex items-center gap-2.5 mb-5">
              <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-cyan-300" />
              </div>
              <h2 className="text-base font-semibold text-zinc-50 tracking-tight">Study Time</h2>
            </div>
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={analytics.study_time_trend.map((d) => ({
                    date: shortDate(d.date),
                    minutes: d.duration_minutes ?? 0,
                    questions: d.questions,
                  }))}
                  margin={{ left: 0, right: 16, top: 8, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" />
                  <YAxis tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" />
                  <Tooltip
                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                    formatter={(value, name) => [name === 'minutes' ? `${value} min` : value, name === 'minutes' ? 'Study time' : 'Questions']}
                  />
                  <Line type="monotone" dataKey="minutes" stroke="#22d3ee" strokeWidth={2} dot={{ r: 3, fill: '#22d3ee' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card padding="lg">
            <div className="flex items-center gap-2.5 mb-5">
              <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
                <Target className="w-5 h-5 text-blue-300" />
              </div>
              <h2 className="text-base font-semibold text-zinc-50 tracking-tight">Confidence Over Time</h2>
            </div>
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={analytics.study_time_trend.map((d) => ({
                    date: shortDate(d.date),
                    confidence: Math.round((d.avg_confidence ?? 0) * 100),
                  }))}
                  margin={{ left: 0, right: 16, top: 8, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="date" tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" />
                  <YAxis domain={[0, 100]} tick={{ fill: '#71717a', fontSize: 11 }} stroke="#27272a" unit="%" />
                  <Tooltip
                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8, fontSize: 12 }}
                    formatter={(value) => [`${value}%`, 'Avg confidence']}
                  />
                  <Line type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3, fill: '#3b82f6' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      )}

      {/* Weak Areas & Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Weak Areas */}
        <Card padding="lg">
          <div className="flex items-center gap-2.5 mb-5">
            <div className="w-9 h-9 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
              <AlertCircle className="w-5 h-5 text-red-400" />
            </div>
            <h2 className="text-base font-semibold text-zinc-50 tracking-tight">Areas to Focus On</h2>
          </div>

          {analytics.weak_areas.length === 0 ? (
            <div className="text-center py-8">
              <Award className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
              <p className="text-emerald-400 font-medium">Great job!</p>
              <p className="text-zinc-500 text-sm">No weak areas identified</p>
            </div>
          ) : (
            <div className="space-y-3">
              {analytics.weak_areas.map((area, index) => (
                <div key={index} className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="font-medium text-red-400 capitalize">{area}</span>
                  </div>
                  <p className="text-red-300 text-sm mt-1">
                    Consider reviewing this topic and practicing more problems
                  </p>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Study Recommendations */}
        <Card padding="lg">
          <div className="flex items-center gap-2.5 mb-5">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Zap className="w-5 h-5 text-amber-400" />
            </div>
            <h2 className="text-base font-semibold text-zinc-50 tracking-tight">Study Recommendations</h2>
          </div>

          <div className="space-y-3">
            {analytics.study_recommendations.map((rec, index) => (
              <div key={index} className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-amber-400 mt-0.5" />
                  <p className="text-amber-200 text-sm">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Study Schedule Suggestion */}
      <Card accent padding="lg">
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center">
            <Calendar className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-base font-semibold text-zinc-50 tracking-tight">Suggested Study Schedule</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="bg-zinc-950/50 rounded-lg p-4 border border-zinc-800">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">Today</h3>
            <p className="text-sm text-zinc-400">
              {analytics.weak_areas.length > 0
                ? `Review ${analytics.weak_areas[0]} concepts`
                : 'Great job! Try exploring new topics'
              }
            </p>
          </div>

          <div className="bg-zinc-950/50 rounded-lg p-4 border border-zinc-800">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">This Week</h3>
            <p className="text-sm text-zinc-400">
              Practice problems on your strongest topics to maintain mastery
            </p>
          </div>

          <div className="bg-zinc-950/50 rounded-lg p-4 border border-zinc-800">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">Next Steps</h3>
            <p className="text-sm text-zinc-400">
              {analytics.total_questions < 50
                ? 'Ask more questions to get better insights'
                : 'Consider taking a practice exam'
              }
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
