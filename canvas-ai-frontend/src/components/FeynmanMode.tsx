import { useState } from 'react'
import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle, AlertTriangle, XCircle, Sparkles, RotateCcw, BookOpen } from 'lucide-react'
import { feynmanEvaluate, type FeynmanResult } from '@/lib/api'
import { showError } from '@/lib/toast'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Input, Textarea } from '@/components/ui/Input'

interface FeynmanModeProps {
  courseId: string
  userId: string
}

const VERDICT_TONE: Record<FeynmanResult['verdict'], { ring: string; text: string; label: string; blurb: string }> = {
  solid: { ring: '#34d399', text: 'text-emerald-400', label: 'Solid', blurb: 'You can teach this.' },
  partial: { ring: '#fbbf24', text: 'text-amber-400', label: 'Partial', blurb: 'Close — a few gaps to close.' },
  shaky: { ring: '#fb7185', text: 'text-rose-400', label: 'Shaky', blurb: "Let's shore up the fundamentals." },
}

export default function FeynmanMode({ courseId, userId }: FeynmanModeProps) {
  const [concept, setConcept] = useState('')
  const [explanation, setExplanation] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<FeynmanResult | null>(null)
  const invalidateProgress = useInvalidateProgress(courseId)

  const submit = async () => {
    if (!concept.trim() || !explanation.trim() || !courseId) return
    setLoading(true)
    try {
      setResult(await feynmanEvaluate(courseId, concept.trim(), explanation.trim(), userId))
      // Grading seeds review items + mastery data — refresh progress views.
      invalidateProgress()
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to grade explanation')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setResult(null)
    setExplanation('')
  }

  const wordCount = explanation.trim() ? explanation.trim().split(/\s+/).length : 0

  if (result) {
    const tone = VERDICT_TONE[result.verdict]
    return (
      <div className="h-full overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-5 py-8">
          <AnimatePresence mode="wait">
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: 'easeOut' }}
              className="space-y-4"
            >
              <Card accent className="flex items-center gap-6">
                <ScoreRing score={result.score_pct} ring={tone.ring} />
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold text-zinc-50 tracking-tight">
                    {concept} — <span className={tone.text}>{tone.label}</span>
                  </h3>
                  <p className="text-sm text-zinc-400">{tone.blurb}</p>
                  <p className="text-sm text-zinc-300 mt-2 leading-relaxed">{result.summary}</p>
                  {result.review_items_added > 0 && (
                    <Badge tone="warning" icon={<BookOpen />} className="mt-2.5">
                      +{result.review_items_added} gap{result.review_items_added === 1 ? '' : 's'} seeded
                      into your spaced review
                    </Badge>
                  )}
                </div>
              </Card>

              {result.strengths.length > 0 && (
                <Section
                  title="What you nailed"
                  icon={<CheckCircle className="w-4 h-4 text-emerald-400" />}
                  items={result.strengths}
                  tone="text-emerald-300"
                  dot="bg-emerald-400/60"
                  delay={0.05}
                />
              )}
              {result.gaps.length > 0 && (
                <Section
                  title="What you missed"
                  icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}
                  items={result.gaps}
                  tone="text-amber-300"
                  dot="bg-amber-400/60"
                  delay={0.12}
                />
              )}
              {result.misconceptions.length > 0 && (
                <Section
                  title="What to rethink"
                  icon={<XCircle className="w-4 h-4 text-rose-400" />}
                  items={result.misconceptions}
                  tone="text-rose-300"
                  dot="bg-rose-400/60"
                  delay={0.19}
                />
              )}

              <Button onClick={reset} leftIcon={<RotateCcw className="w-4 h-4" />}>
                Explain it again
              </Button>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-5 py-8">
        <Card className="space-y-5">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-400/20 glow-brand-sm flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-4.5 h-4.5 text-cyan-300" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-zinc-50">Explain it in your own words</h3>
              <p className="text-sm text-zinc-400 mt-1 leading-relaxed">
                If you can teach it, you understand it. Explain a concept like you would to a classmate —
                I'll grade it against your course material and surface your blind spots.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <Field label="Pick a concept" step={1}>
              <Input
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                placeholder="e.g. Topological sort"
              />
            </Field>

            <Field label="Explain it" step={2}>
              <Textarea
                value={explanation}
                onChange={(e) => setExplanation(e.target.value)}
                placeholder="Explain it as if teaching a classmate who's never seen it…"
                rows={8}
                className="resize-none"
              />
              {wordCount > 0 && (
                <p className="text-[11px] text-zinc-500 mt-1.5 text-right">{wordCount} words</p>
              )}
            </Field>
          </div>

          <Button
            onClick={() => void submit()}
            loading={loading}
            disabled={loading || !concept.trim() || !explanation.trim()}
            leftIcon={!loading ? <Sparkles className="w-4 h-4" /> : undefined}
            className="w-full"
          >
            {loading ? 'Grading your explanation…' : 'Grade my explanation'}
          </Button>
        </Card>
      </div>
    </div>
  )
}

function ScoreRing({ score, ring }: { score: number; ring: string }) {
  const circ = 2 * Math.PI * 36
  const offset = circ * (1 - score / 100)
  return (
    <div className="relative w-28 h-28 flex-shrink-0">
      <svg className="w-28 h-28 -rotate-90" viewBox="0 0 88 88">
        <circle cx="44" cy="44" r="36" fill="none" stroke="#252e42" strokeWidth="8" />
        <motion.circle
          cx="44"
          cy="44"
          r="36"
          fill="none"
          stroke={ring}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-gradient-brand leading-none">{score}</span>
        <span className="text-[10px] uppercase tracking-widest text-zinc-500 mt-0.5">score</span>
      </div>
    </div>
  )
}

function Field({ label, step, children }: { label: string; step: number; children: ReactNode }) {
  return (
    <div>
      <label className="flex items-center gap-2 text-xs font-medium text-zinc-400 mb-1.5">
        <span className="w-4 h-4 rounded-full bg-gradient-brand-soft border border-cyan-400/20 text-[10px] text-cyan-300 flex items-center justify-center">
          {step}
        </span>
        {label}
      </label>
      {children}
    </div>
  )
}

function Section({
  title,
  icon,
  items,
  tone,
  dot,
  delay,
}: {
  title: string
  icon: ReactNode
  items: string[]
  tone: string
  dot: string
  delay: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay }}
    >
      <Card padding="sm">
        <h4 className="text-sm font-semibold text-zinc-100 mb-2.5 flex items-center gap-2">
          {icon}
          {title}
          <span className="ml-auto text-[11px] font-normal text-zinc-500">{items.length}</span>
        </h4>
        <ul className="space-y-2">
          {items.map((it, i) => (
            <li key={i} className={`text-sm ${tone} flex gap-2.5 items-start`}>
              <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
              <span className="leading-relaxed">{it}</span>
            </li>
          ))}
        </ul>
      </Card>
    </motion.div>
  )
}
