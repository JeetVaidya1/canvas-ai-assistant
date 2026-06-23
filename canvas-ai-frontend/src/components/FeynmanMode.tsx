import { useState } from 'react'
import { Loader2, CheckCircle, AlertTriangle, XCircle, Sparkles, RotateCcw } from 'lucide-react'
import { feynmanEvaluate, type FeynmanResult } from '@/lib/api'
import { showError } from '@/lib/toast'

interface FeynmanModeProps {
  courseId: string
  userId: string
}

const VERDICT_TONE: Record<FeynmanResult['verdict'], { ring: string; text: string; label: string }> = {
  solid: { ring: '#10b981', text: 'text-emerald-400', label: 'Solid' },
  partial: { ring: '#f59e0b', text: 'text-amber-400', label: 'Partial' },
  shaky: { ring: '#ef4444', text: 'text-red-400', label: 'Shaky' },
}

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
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 flex items-center gap-5">
          <div className="relative w-24 h-24 flex-shrink-0">
            <svg className="w-24 h-24 -rotate-90" viewBox="0 0 88 88">
              <circle cx="44" cy="44" r="36" fill="none" stroke="#27272a" strokeWidth="8" />
              <circle cx="44" cy="44" r="36" fill="none" stroke={tone.ring} strokeWidth="8" strokeLinecap="round"
                strokeDasharray={circ} strokeDashoffset={offset} style={{ transition: 'stroke-dashoffset 0.7s ease' }} />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`text-xl font-bold ${tone.text}`}>{result.score_pct}%</span>
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
        </div>

        {result.strengths.length > 0 && (
          <Section title="What you nailed" icon={<CheckCircle className="w-4 h-4 text-emerald-400" />} items={result.strengths} tone="text-emerald-300" />
        )}
        {result.gaps.length > 0 && (
          <Section title="What you missed" icon={<AlertTriangle className="w-4 h-4 text-amber-400" />} items={result.gaps} tone="text-amber-300" />
        )}
        {result.misconceptions.length > 0 && (
          <Section title="What to rethink" icon={<XCircle className="w-4 h-4 text-red-400" />} items={result.misconceptions} tone="text-red-300" />
        )}

        <button onClick={reset} className="bg-violet-600 text-white px-4 py-2 rounded-lg hover:bg-violet-500 text-sm font-medium flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> Explain another concept
        </button>
      </div>
    )
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
      <div>
        <h3 className="text-sm font-medium text-zinc-100 mb-1">Explain it back</h3>
        <p className="text-xs text-zinc-500">Teach a concept in your own words. I'll grade it against your course material and find your blind spots.</p>
      </div>
      <input
        value={concept}
        onChange={(e) => setConcept(e.target.value)}
        placeholder="Concept (e.g. Topological sort)"
        className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
      />
      <textarea
        value={explanation}
        onChange={(e) => setExplanation(e.target.value)}
        placeholder="Explain it as if teaching a classmate…"
        rows={7}
        className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm resize-none"
      />
      <button
        onClick={() => void submit()}
        disabled={loading || !concept.trim() || !explanation.trim()}
        className="bg-violet-600 text-white px-4 py-2 rounded-lg hover:bg-violet-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
      >
        {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Grading…</> : <><Sparkles className="w-4 h-4" /> Grade my explanation</>}
      </button>
    </div>
  )
}

function Section({ title, icon, items, tone }: { title: string; icon: React.ReactNode; items: string[]; tone: string }) {
  return (
    <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-4">
      <h4 className="text-sm font-medium text-zinc-200 mb-2 flex items-center gap-2">{icon}{title}</h4>
      <ul className="space-y-1.5">
        {items.map((it, i) => (
          <li key={i} className={`text-sm ${tone} flex gap-2`}><span className="text-zinc-600">•</span><span>{it}</span></li>
        ))}
      </ul>
    </div>
  )
}
