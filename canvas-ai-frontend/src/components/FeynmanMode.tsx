import { useState } from 'react'
import { CheckCircle, AlertTriangle, XCircle, Sparkles, RotateCcw } from 'lucide-react'
import { feynmanEvaluate, type FeynmanResult } from '@/lib/api'
import { showError } from '@/lib/toast'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface FeynmanModeProps {
  courseId: string
  userId: string
}

const VERDICT_TONE: Record<FeynmanResult['verdict'], { ring: string; text: string; label: string }> = {
  solid: { ring: '#10b981', text: 'text-emerald-400', label: 'Solid' },
  partial: { ring: '#f59e0b', text: 'text-amber-400', label: 'Partial' },
  shaky: { ring: '#ef4444', text: 'text-red-400', label: 'Shaky' },
}

const inputClass =
  'w-full px-3 py-2.5 bg-zinc-800/70 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-500 ' +
  'focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-sm transition-colors'

export default function FeynmanMode({ courseId, userId }: FeynmanModeProps) {
  const [concept, setConcept] = useState('')
  const [explanation, setExplanation] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<FeynmanResult | null>(null)

  const submit = async () => {
    if (!concept.trim() || !explanation.trim() || !courseId) return
    setLoading(true)
    try {
      setResult(await feynmanEvaluate(courseId, concept.trim(), explanation.trim(), userId))
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

  if (result) {
    const tone = VERDICT_TONE[result.verdict]
    const circ = 2 * Math.PI * 36
    const offset = circ * (1 - result.score_pct / 100)
    return (
      <div className="space-y-4">
        <Card accent className="flex items-center gap-5">
          <div className="relative w-24 h-24 flex-shrink-0">
            <svg className="w-24 h-24 -rotate-90" viewBox="0 0 88 88">
              <circle cx="44" cy="44" r="36" fill="none" stroke="#27272a" strokeWidth="8" />
              <circle cx="44" cy="44" r="36" fill="none" stroke={tone.ring} strokeWidth="8" strokeLinecap="round"
                strokeDasharray={circ} strokeDashoffset={offset} style={{ transition: 'stroke-dashoffset 0.7s ease' }} />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold text-gradient-brand">{result.score_pct}%</span>
            </div>
          </div>
          <div>
            <h3 className="text-base font-semibold text-zinc-100">{concept} — <span className={tone.text}>{tone.label}</span></h3>
            <p className="text-sm text-zinc-400 mt-1">{result.summary}</p>
            {result.review_items_added > 0 && (
              <p className="text-xs text-amber-400 mt-2">
                + {result.review_items_added} gap{result.review_items_added === 1 ? '' : 's'} added to your review queue
              </p>
            )}
          </div>
        </Card>

        {result.strengths.length > 0 && (
          <Section title="What you nailed" icon={<CheckCircle className="w-4 h-4 text-emerald-400" />} items={result.strengths} tone="text-emerald-300" />
        )}
        {result.gaps.length > 0 && (
          <Section title="What you missed" icon={<AlertTriangle className="w-4 h-4 text-amber-400" />} items={result.gaps} tone="text-amber-300" />
        )}
        {result.misconceptions.length > 0 && (
          <Section title="What to rethink" icon={<XCircle className="w-4 h-4 text-red-400" />} items={result.misconceptions} tone="text-red-300" />
        )}

        <Button onClick={reset} leftIcon={<RotateCcw className="w-4 h-4" />}>
          Explain another concept
        </Button>
      </div>
    )
  }

  return (
    <Card className="space-y-4">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center flex-shrink-0">
          <Sparkles className="w-4 h-4 text-cyan-300" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">Explain it back</h3>
          <p className="text-xs text-zinc-500 mt-0.5">Teach a concept in your own words. I'll grade it against your course material and find your blind spots.</p>
        </div>
      </div>
      <input
        value={concept}
        onChange={(e) => setConcept(e.target.value)}
        placeholder="Concept (e.g. Topological sort)"
        className={inputClass}
      />
      <textarea
        value={explanation}
        onChange={(e) => setExplanation(e.target.value)}
        placeholder="Explain it as if teaching a classmate…"
        rows={7}
        className={`${inputClass} resize-none`}
      />
      <Button
        onClick={() => void submit()}
        loading={loading}
        disabled={loading || !concept.trim() || !explanation.trim()}
        leftIcon={!loading ? <Sparkles className="w-4 h-4" /> : undefined}
      >
        {loading ? 'Grading…' : 'Grade my explanation'}
      </Button>
    </Card>
  )
}

function Section({ title, icon, items, tone }: { title: string; icon: React.ReactNode; items: string[]; tone: string }) {
  return (
    <Card padding="sm">
      <h4 className="text-sm font-medium text-zinc-200 mb-2 flex items-center gap-2">{icon}{title}</h4>
      <ul className="space-y-1.5">
        {items.map((it, i) => (
          <li key={i} className={`text-sm ${tone} flex gap-2`}><span className="text-cyan-500/60">•</span><span>{it}</span></li>
        ))}
      </ul>
    </Card>
  )
}
