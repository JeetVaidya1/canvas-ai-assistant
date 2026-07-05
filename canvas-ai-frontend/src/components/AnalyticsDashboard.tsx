// src/components/AnalyticsDashboard.tsx
import { useMemo } from 'react'
import { motion } from 'motion/react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
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
  Calendar,
  Flame,
  Network,
  ArrowRight,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import {
  type Readiness,
  type ConceptGraph,
  type LearningAnalytics,
} from '@/lib/api'
import { useLearningAnalytics, useConceptGraph } from '@/hooks/useAnalytics'
import { useReadiness } from '@/hooks/useReadiness'
import { Card, PageHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import CountUp from '@/components/ui/CountUp'
import ReviewPanel from './ReviewPanel'
import ErrorInline from './shared/ErrorInline'

type AnalyticsData = LearningAnalytics

// rose · amber · indigo · emerald — semantic mastery ramp on the dark canvas
const MASTERY_COLORS = ['#fb7185', '#fbbf24', '#818cf8', '#34d399']

function masteryColor(level: number): string {
  if (level >= 0.8) return MASTERY_COLORS[3]
  if (level >= 0.7) return MASTERY_COLORS[2]
  if (level >= 0.5) return MASTERY_COLORS[1]
  return MASTERY_COLORS[0]
}

/** Mastery color from a 0–100 percentage (concept graph + topic bars). */
function masteryColorPct(pct: number): string {
  return masteryColor(pct / 100)
}

function shortDate(iso: string): string {
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

/** Section header used across the dashboard tiles for a consistent hierarchy. */
function SectionHead({
  icon: Icon,
  title,
  chip = 'bg-cyan-500/12 border-cyan-400/20',
  tint = 'text-cyan-300',
  hint,
}: {
  icon: typeof Flame
  title: string
  chip?: string
  tint?: string
  hint?: string
}) {
  return (
    <div className="mb-5 flex items-start gap-2.5">
      <div className={`w-9 h-9 rounded-xl border flex items-center justify-center flex-shrink-0 ${chip}`}>
        <Icon className={`w-5 h-5 ${tint}`} />
      </div>
      <div className="min-w-0">
        <h2 className="text-base font-semibold text-zinc-50 tracking-tight leading-tight">{title}</h2>
        {hint && <p className="text-xs text-zinc-400 mt-0.5">{hint}</p>}
      </div>
    </div>
  )
}

/** Tasteful, readable empty state for tiles whose data often reads 0. */
function TileEmpty({ icon: Icon, line, hint }: { icon: typeof Brain; line: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-10 px-4">
      <div className="w-12 h-12 rounded-2xl bg-white/[0.04] border border-white/10 flex items-center justify-center mb-3">
        <Icon className="w-6 h-6 text-zinc-400" />
      </div>
      <p className="text-sm text-zinc-300">{line}</p>
      <p className="text-xs text-zinc-500 mt-1 max-w-[16rem]">{hint}</p>
    </div>
  )
}

interface StatCardProps {
  icon: typeof Flame
  tint: string
  chip: string
  label: string
  value: number
  suffix?: string
  unit: string
}

function StatCard({ icon: Icon, tint, chip, label, value, suffix = '', unit }: StatCardProps) {
  return (
    <Card accent elevation={1}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-zinc-400">{label}</p>
          <p className="text-3xl font-bold text-zinc-50 mt-1.5 leading-none tabular-nums">
            <CountUp to={value} />{suffix}
          </p>
          <p className="text-xs text-zinc-500 mt-1.5">{unit}</p>
        </div>
        <div className={`w-10 h-10 rounded-xl border flex items-center justify-center ${chip}`}>
          <Icon className={`w-5 h-5 ${tint}`} />
        </div>
      </div>
    </Card>
  )
}

interface AnalyticsDashboardProps {
  courseId: string
  userId: string
}

function readinessTone(score: number): { ring: string; text: string; label: string } {
  if (score >= 70) return { ring: '#34d399', text: 'text-emerald-300', label: 'On track' }
  if (score >= 40) return { ring: '#fbbf24', text: 'text-amber-300', label: 'Getting there' }
  return { ring: '#fb7185', text: 'text-rose-300', label: 'At risk' }
}

function ReadinessHero({ readiness }: { readiness: Readiness }) {
  const score = Math.round(readiness.score_pct)
  const tone = readinessTone(score)
  const r = 48
  const circ = 2 * Math.PI * r
  return (
    <Card accent padding="lg" elevation={2} className="flex flex-col md:flex-row items-center gap-6">
      <div className="relative w-[116px] h-[116px] flex-shrink-0">
        <svg className="w-[116px] h-[116px] -rotate-90" viewBox="0 0 116 116">
          <circle cx="58" cy="58" r={r} fill="none" stroke="#222228" strokeWidth="9" />
          <motion.circle
            cx="58" cy="58" r={r} fill="none" stroke={tone.ring} strokeWidth="9" strokeLinecap="round"
            strokeDasharray={circ}
            initial={{ strokeDashoffset: circ }}
            animate={{ strokeDashoffset: circ * (1 - score / 100) }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
            style={{ filter: `drop-shadow(0 0 6px ${tone.ring}55)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${tone.text}`}>{score}%</span>
          <span className="text-[10px] text-zinc-500 uppercase tracking-widest mt-0.5">ready</span>
        </div>
      </div>
      <div className="flex-1 text-center md:text-left">
        <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Exam readiness</p>
        <div className="flex items-center justify-center md:justify-start gap-2 mb-1.5">
          <h2 className={`text-xl font-semibold tracking-tight ${tone.text}`}>{tone.label}</h2>
        </div>
        <p className="text-sm text-zinc-400 mb-3">
          {readiness.has_past_papers
            ? 'Weighted by how often each topic shows up on your past papers.'
            : 'Based on your topic mastery. Upload a past paper to weight by what’s actually tested.'}
          {readiness.confidence === 'low' && ' Study more to sharpen this estimate.'}
        </p>
        {readiness.gaps.length > 0 ? (
          <div className="flex flex-wrap gap-2 justify-center md:justify-start">
            <span className="text-xs text-zinc-400 self-center">Biggest gaps:</span>
            {readiness.gaps.map((g) => (
              <span key={g} className="text-xs text-amber-200 bg-amber-500/10 border border-amber-500/25 rounded-full px-2.5 py-0.5">
                {g}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-xs text-emerald-300">No major gaps — keep reviewing to hold your edge.</p>
        )}
      </div>
    </Card>
  )
}

/** Per-topic mastery breakdown — uses readiness.by_topic. Shows where the
 *  readiness score actually comes from, with legible labels + gradient fills. */
function TopicMastery({ readiness }: { readiness: Readiness }) {
  const topics = (readiness.by_topic ?? []).filter((t) => t.has_data).slice(0, 12)
  return (
    <Card padding="lg" className="h-full">
      <SectionHead
        icon={TrendingUp}
        title="Topic mastery"
        hint="Where your readiness score comes from"
      />
      {topics.length === 0 ? (
        <TileEmpty
          icon={Brain}
          line="No topic mastery yet"
          hint="Ask questions or take a quiz to start scoring topics."
        />
      ) : (
        <div className="space-y-3.5">
          {topics.map((t) => (
            <div key={t.topic}>
              <div className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-zinc-200 truncate pr-3">{t.topic}</span>
                <span className="text-zinc-100 tabular-nums font-semibold">{Math.round(t.mastery_pct)}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-white/[0.06] overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-brand"
                  initial={{ width: 0 }}
                  whileInView={{ width: `${Math.min(100, Math.max(2, t.mastery_pct))}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

/** ── Concept prerequisite graph ──────────────────────────────────────
 *  A real node/edge diagram on navy. Nodes are laid out in dependency
 *  layers (topological-ish): prerequisites on the left, dependents on the
 *  right, mastery color-coded. This is a product differentiator, so it
 *  gets a wide, prominent tile with a legend. */
interface GraphNode {
  concept: string
  mastery_pct: number
  has_data: boolean
}

function layoutGraph(graph: ConceptGraph) {
  const nodes = graph.concepts
  if (nodes.length === 0) {
    return {
      positioned: [] as Array<{ node: GraphNode; x: number; y: number }>,
      edges: [] as Array<{ from: { x: number; y: number }; to: { x: number; y: number }; prerequisite: string; concept: string }>,
      width: 1000,
      height: 0,
    }
  }

  // Depth = longest prerequisite chain ending at a node (layer index).
  const byName = new Map(nodes.map((n) => [n.concept, n]))
  const incoming = new Map<string, string[]>()
  nodes.forEach((n) => incoming.set(n.concept, []))
  graph.edges.forEach((e) => {
    if (byName.has(e.prerequisite) && byName.has(e.concept)) {
      incoming.get(e.concept)!.push(e.prerequisite)
    }
  })

  const depthCache = new Map<string, number>()
  const depthOf = (name: string, seen: Set<string>): number => {
    if (depthCache.has(name)) return depthCache.get(name)!
    if (seen.has(name)) return 0 // cycle guard
    seen.add(name)
    const preds = incoming.get(name) ?? []
    const d = preds.length === 0 ? 0 : 1 + Math.max(...preds.map((p) => depthOf(p, seen)))
    seen.delete(name)
    depthCache.set(name, d)
    return d
  }

  const layers = new Map<number, GraphNode[]>()
  nodes.forEach((n) => {
    const d = depthOf(n.concept, new Set())
    if (!layers.has(d)) layers.set(d, [])
    layers.get(d)!.push(n)
  })

  const maxDepth = Math.max(0, ...[...layers.keys()])
  const W = 1000
  const colGap = maxDepth === 0 ? 0 : W / (maxDepth + 1)
  const rowGap = 92
  const positioned: Array<{ node: GraphNode; x: number; y: number }> = []
  const posByName = new Map<string, { x: number; y: number }>()

  for (let d = 0; d <= maxDepth; d++) {
    const layer = layers.get(d) ?? []
    const colX = maxDepth === 0 ? W / 2 : colGap * d + colGap / 2
    layer.forEach((node, i) => {
      const y = (i + 1) * rowGap
      const x = colX
      positioned.push({ node, x, y })
      posByName.set(node.concept, { x, y })
    })
  }

  const edges = graph.edges
    .map((e) => {
      const from = posByName.get(e.prerequisite)
      const to = posByName.get(e.concept)
      if (!from || !to) return null
      return { from, to, prerequisite: e.prerequisite, concept: e.concept }
    })
    .filter((e): e is NonNullable<typeof e> => e !== null)

  const maxRows = Math.max(1, ...[...layers.values()].map((l) => l.length))
  const height = (maxRows + 1) * rowGap
  return { positioned, edges, width: W, height }
}

function ConceptGraphTile({ graph }: { graph: ConceptGraph | null }) {
  const layout = useMemo(() => (graph ? layoutGraph(graph) : null), [graph])
  const hasData = graph && graph.concepts.length > 0

  return (
    <Card accent padding="lg" elevation={2}>
      <div className="flex items-start justify-between gap-3 mb-5">
        <SectionHead
          icon={Network}
          title="Concept map"
          hint="How concepts build on each other — fix the upstream gaps first"
        />
        {hasData && (
          <div className="hidden sm:flex items-center gap-3 flex-shrink-0 pt-1">
            {[
              { c: MASTERY_COLORS[3], l: 'Strong' },
              { c: MASTERY_COLORS[1], l: 'Shaky' },
              { c: MASTERY_COLORS[0], l: 'Weak' },
            ].map((it) => (
              <span key={it.l} className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: it.c }} />
                {it.l}
              </span>
            ))}
          </div>
        )}
      </div>

      {!hasData || !layout || layout.positioned.length === 0 ? (
        <TileEmpty
          icon={Network}
          line="Concept map not built yet"
          hint="Study a few topics — we’ll map how your concepts depend on each other."
        />
      ) : (
        <div className="overflow-x-auto -mx-2 px-2">
          <svg
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            className="w-full min-w-[640px]"
            style={{ height: Math.min(420, layout.height) }}
          >
            {/* Edges */}
            {layout.edges.map((e, i) => {
              const midX = (e.from.x + e.to.x) / 2
              const d = `M ${e.from.x} ${e.from.y} C ${midX} ${e.from.y}, ${midX} ${e.to.y}, ${e.to.x} ${e.to.y}`
              return (
                <motion.path
                  key={i}
                  d={d}
                  fill="none"
                  stroke="#38445e"
                  strokeWidth={1.5}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.7, delay: 0.15 + i * 0.03, ease: 'easeOut' }}
                />
              )
            })}
            {/* Nodes */}
            {layout.positioned.map(({ node, x, y }, i) => {
              const color = node.has_data ? masteryColorPct(node.mastery_pct) : '#3a4358'
              const label = node.concept.length > 22 ? node.concept.slice(0, 21) + '…' : node.concept
              const w = Math.max(120, label.length * 7.6 + 28)
              return (
                <motion.g
                  key={node.concept}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.35, delay: 0.1 + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
                >
                  <rect
                    x={x - w / 2}
                    y={y - 17}
                    width={w}
                    height={34}
                    rx={9}
                    fill="#19202f"
                    stroke={color}
                    strokeWidth={1.5}
                    style={{ filter: `drop-shadow(0 2px 6px rgba(0,0,0,0.4))` }}
                  />
                  <circle cx={x - w / 2 + 14} cy={y} r={4} fill={color} />
                  <text
                    x={x - w / 2 + 26}
                    y={y + 4}
                    fontSize={12}
                    fill="#e6e6ec"
                    fontWeight={500}
                  >
                    {label}
                  </text>
                  {node.has_data && (
                    <text x={x + w / 2 - 10} y={y + 4} fontSize={11} fill={color} textAnchor="end" fontWeight={600}>
                      {Math.round(node.mastery_pct)}%
                    </text>
                  )}
                </motion.g>
              )
            })}
          </svg>
        </div>
      )}

      {/* Foundation-first blockers, surfaced under the graph it explains. */}
      {graph && graph.blockers.length > 0 && (
        <div className="mt-5 pt-4 border-t border-white/[0.06]">
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-300/90 mb-3">Fix the foundation first</p>
          <div className="space-y-2">
            {graph.blockers.slice(0, 5).map((b, i) => (
              <div key={i} className="flex items-center flex-wrap gap-2 text-sm">
                <span className="text-amber-200 bg-amber-500/10 border border-amber-500/25 rounded px-2 py-0.5 text-xs">
                  {b.prerequisite} <span className="text-amber-300/70">({Math.round(b.prerequisite_pct)}%)</span>
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-zinc-600" />
                <span className="text-zinc-300 text-xs">
                  {b.concept} <span className="text-zinc-500">({Math.round(b.concept_pct)}%)</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

/** Weak areas + recommendations — stacked in a single right-rail tile. */
function FocusTile({ analytics }: { analytics: AnalyticsData }) {
  return (
    <div className="flex flex-col gap-4 h-full">
      <Card padding="lg" className="flex-1">
        <SectionHead
          icon={AlertCircle}
          title="Areas to focus on"
          chip="bg-rose-500/10 border-rose-500/20"
          tint="text-rose-400"
        />
        {analytics.weak_areas.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-6">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-400/20 flex items-center justify-center mb-3">
              <Award className="w-6 h-6 text-emerald-300" />
            </div>
            <p className="text-emerald-300 font-medium">No weak areas</p>
            <p className="text-zinc-400 text-sm mt-0.5">Keep practicing to hold your edge.</p>
          </div>
        ) : (
          <div className="space-y-2.5">
            {analytics.weak_areas.map((area, index) => (
              <div key={index} className="bg-rose-500/10 border border-rose-500/25 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                  <span className="font-medium text-rose-300 capitalize">{area}</span>
                </div>
                <p className="text-rose-200/80 text-sm mt-1">Review this topic and practice more problems.</p>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card padding="lg" className="flex-1">
        <SectionHead
          icon={Zap}
          title="Recommendations"
          chip="bg-amber-500/10 border-amber-500/20"
          tint="text-amber-400"
        />
        {analytics.study_recommendations.length === 0 ? (
          <TileEmpty
            icon={Zap}
            line="No recommendations yet"
            hint="Study a bit more and we’ll suggest your next moves."
          />
        ) : (
          <div className="space-y-2.5">
            {analytics.study_recommendations.map((rec, index) => (
              <div key={index} className="bg-amber-500/10 border border-amber-500/25 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <p className="text-amber-100/90 text-sm">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

/** A single dark-styled Recharts line tile (study time / confidence). */
function TrendTile({
  icon,
  title,
  chip,
  tint,
  data,
  dataKey,
  stroke,
  unit,
  domain,
  formatter,
}: {
  icon: typeof Calendar
  title: string
  chip: string
  tint: string
  data: Array<Record<string, number | string>>
  dataKey: string
  stroke: string
  unit?: string
  domain?: [number, number]
  formatter: (value: number | string) => [string, string]
}) {
  return (
    <Card padding="lg">
      <SectionHead icon={icon} title={title} chip={chip} tint={tint} />
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2738" />
            <XAxis dataKey="date" tick={{ fill: '#9aa6bd', fontSize: 11 }} stroke="#1f2738" />
            <YAxis domain={domain} unit={unit} tick={{ fill: '#9aa6bd', fontSize: 11 }} stroke="#1f2738" />
            <Tooltip
              contentStyle={{ background: '#1f2738', border: '1px solid #252e42', borderRadius: 8, fontSize: 12, color: '#f3f6fc' }}
              formatter={(value) => formatter(value as number | string)}
            />
            <Line type="monotone" dataKey={dataKey} stroke={stroke} strokeWidth={2} dot={{ r: 3, fill: stroke }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

export default function AnalyticsDashboard({ courseId, userId }: AnalyticsDashboardProps) {
  // Analytics + readiness gate the page; the concept graph loads separately
  // (its first call builds the graph server-side and is slow — non-blocking).
  const analyticsQuery = useLearningAnalytics(courseId, userId)
  const readinessQuery = useReadiness(courseId, userId)
  const graphQuery = useConceptGraph(courseId, userId)

  const analytics = analyticsQuery.data ?? null
  const readiness = readinessQuery.data ?? null
  const graph = graphQuery.data ?? null
  const loading = analyticsQuery.isPending || readinessQuery.isPending

  const refreshAll = () => {
    void analyticsQuery.refetch()
    void readinessQuery.refetch()
    void graphQuery.refetch()
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-white/[0.04] rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-white/[0.04] rounded-xl"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (analyticsQuery.isError) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <ErrorInline
          message="Couldn't load your analytics."
          onRetry={refreshAll}
        />
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <Card accent padding="none" elevation={2} className="py-16 px-8 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">No analytics data yet</h3>
          <p className="text-sm text-zinc-400">Start studying to see your progress!</p>
        </Card>
      </div>
    )
  }

  const hasTrend = analytics.study_time_trend.length > 0
  const timeData = analytics.study_time_trend.map((d) => ({
    date: shortDate(d.date),
    minutes: d.duration_minutes ?? 0,
  }))
  const confData = analytics.study_time_trend.map((d) => ({
    date: shortDate(d.date),
    confidence: Math.round((d.avg_confidence ?? 0) * 100),
  }))

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <PageHeader
        eyebrow="Analytics"
        title="Learning Analytics"
        subtitle="Track your progress and identify areas for improvement"
        actions={
          <Button variant="secondary" onClick={refreshAll} leftIcon={<TrendingUp className="w-4 h-4" />}>
            Refresh
          </Button>
        }
      />

      {/* ── HERO: exam readiness, full-width ─────────────────────────── */}
      {readinessQuery.isError ? (
        <ErrorInline
          message="Couldn't load your exam readiness."
          onRetry={() => void readinessQuery.refetch()}
        />
      ) : (
        readiness && <ReadinessHero readiness={readiness} />
      )}

      {/* ── Stat strip ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Flame}
          tint="text-cyan-300"
          chip="bg-cyan-500/12 border-cyan-400/20"
          label="Study Streak"
          value={analytics.study_streak}
          unit="days"
        />
        <StatCard
          icon={BookOpen}
          tint="text-sky-300"
          chip="bg-blue-500/12 border-blue-400/20"
          label="Questions Asked"
          value={analytics.total_questions}
          unit="total"
        />
        <StatCard
          icon={Target}
          tint="text-emerald-300"
          chip="bg-emerald-500/12 border-emerald-400/20"
          label="Avg Confidence"
          value={Math.round(analytics.avg_confidence * 100)}
          suffix="%"
          unit="score"
        />
        <StatCard
          icon={Brain}
          tint="text-sky-300"
          chip="bg-sky-500/12 border-sky-400/20"
          label="Topics Studied"
          value={analytics.topics_progress.length}
          unit="concepts"
        />
      </div>

      {/* ── Mistake-driven review queue (hidden when nothing is due) ──── */}
      <ReviewPanel courseId={courseId} userId={userId} />

      {/* ── Concept prerequisite graph — wide, prominent differentiator ─ */}
      {graphQuery.isError ? (
        <Card padding="lg">
          <SectionHead
            icon={Network}
            title="Concept map"
            hint="How concepts build on each other — fix the upstream gaps first"
          />
          <ErrorInline
            message="Couldn't load your concept map."
            onRetry={() => void graphQuery.refetch()}
          />
        </Card>
      ) : (
        <ConceptGraphTile graph={graph} />
      )}

      {/* ── BENTO: Topic mastery (tall) + Focus rail (weak / recs) ────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {readiness ? (
          <TopicMastery readiness={readiness} />
        ) : (
          <Card padding="lg" className="h-full">
            <SectionHead icon={TrendingUp} title="Topic mastery" hint="Where your readiness score comes from" />
            <TileEmpty icon={Brain} line="No topic mastery yet" hint="Ask questions or take a quiz to start scoring topics." />
          </Card>
        )}
        <FocusTile analytics={analytics} />
      </div>

      {/* ── BENTO: study-time & confidence trends ────────────────────── */}
      {hasTrend ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrendTile
            icon={Calendar}
            title="Study time"
            chip="bg-cyan-500/12 border-cyan-400/20"
            tint="text-cyan-300"
            data={timeData}
            dataKey="minutes"
            stroke="#22d3ee"
            formatter={(value) => [`${value} min`, 'Study time']}
          />
          <TrendTile
            icon={Target}
            title="Confidence over time"
            chip="bg-blue-500/12 border-blue-400/20"
            tint="text-sky-300"
            data={confData}
            dataKey="confidence"
            stroke="#3b82f6"
            unit="%"
            domain={[0, 100]}
            formatter={(value) => [`${value}%`, 'Avg confidence']}
          />
        </div>
      ) : (
        <Card padding="lg">
          <SectionHead icon={Calendar} title="Study activity" chip="bg-cyan-500/12 border-cyan-400/20" tint="text-cyan-300" />
          <TileEmpty icon={Calendar} line="No study activity yet" hint="Your daily study time and confidence trends will appear here." />
        </Card>
      )}

      {/* ── Suggested study schedule ─────────────────────────────────── */}
      <Card accent padding="lg">
        <SectionHead icon={Calendar} title="Suggested study schedule" chip="bg-cyan-500/12 border-cyan-400/20" tint="text-cyan-300" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="bg-white/[0.04] rounded-lg p-4 border border-white/10">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">Today</h3>
            <p className="text-sm text-zinc-300">
              {analytics.weak_areas.length > 0
                ? `Review ${analytics.weak_areas[0]} concepts`
                : 'Great job! Try exploring new topics'}
            </p>
          </div>
          <div className="bg-white/[0.04] rounded-lg p-4 border border-white/10">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">This Week</h3>
            <p className="text-sm text-zinc-300">Practice problems on your strongest topics to maintain mastery</p>
          </div>
          <div className="bg-white/[0.04] rounded-lg p-4 border border-white/10">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gradient-brand mb-2">Next Steps</h3>
            <p className="text-sm text-zinc-300">
              {analytics.total_questions < 50
                ? 'Ask more questions to get better insights'
                : 'Consider taking a practice exam'}
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
