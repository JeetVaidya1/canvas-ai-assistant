// Center-first exam setup: difficulty tiles, question-count segmented control,
// prominent CTA, and the opt-in past-paper upload reveal.
import { useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Timer } from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'
import { ErrorState } from '@/components/ui/States'
import { COUNTS, DIFFICULTIES } from './examMeta'
import { PastPaperPanel } from './PastPaperPanel'
import type { ExamDifficulty, PastPaperAnalysis } from './types'

interface ExamSetupProps {
  difficulty: ExamDifficulty
  onDifficultyChange: (d: ExamDifficulty) => void
  questionCount: number
  onQuestionCountChange: (n: number) => void
  loading: boolean
  genError: string | null
  canGenerate: boolean
  onGenerate: () => void
  onLoadSample: () => void
  uploading: boolean
  analysisSummary: PastPaperAnalysis | null
  onUploadPaper: (file: File) => void
}

export function ExamSetup({
  difficulty,
  onDifficultyChange,
  questionCount,
  onQuestionCountChange,
  loading,
  genError,
  canGenerate,
  onGenerate,
  onLoadSample,
  uploading,
  analysisSummary,
  onUploadPaper,
}: ExamSetupProps) {
  // Past-paper upload is a secondary, opt-in section on the setup screen.
  const [showUpload, setShowUpload] = useState(false)

  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-xl"
      >
        {/* Identity */}
        <div className="mb-8 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand-sm" />
          <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
            Set up your mock exam
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
            A timed simulation built from your materials. Answers and the clock auto-save — close the tab and pick up
            exactly where you left off. AI judges with partial credit and grounded explanations.
          </p>
        </div>

        {/* Difficulty — 4 big selectable tiles */}
        <div className="mb-6">
          <div className="mb-2.5 text-center text-xs font-medium text-zinc-400">Difficulty</div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {DIFFICULTIES.map((d) => {
              const active = difficulty === d.value
              return (
                <button
                  key={d.value}
                  onClick={() => onDifficultyChange(d.value)}
                  aria-pressed={active}
                  className={`group rounded-2xl border px-3 py-4 text-center transition-all ${
                    active
                      ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/25 shadow-[0_8px_24px_-12px_rgba(34,211,238,0.5)]'
                      : 'border-white/10 bg-white/[0.03] hover:border-cyan-400/30 hover:bg-white/[0.05]'
                  }`}
                >
                  <div className={`text-sm font-semibold ${active ? 'text-cyan-200' : 'text-zinc-200'}`}>
                    {d.label}
                  </div>
                  <div className="mt-0.5 text-[10px] text-zinc-500 leading-tight">{d.hint}</div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Question count — segmented control */}
        <div className="mb-8">
          <div className="mb-2.5 text-center text-xs font-medium text-zinc-400">Questions</div>
          <div className="flex gap-1.5 rounded-2xl border border-white/10 bg-white/[0.03] p-1.5">
            {COUNTS.map((c) => {
              const active = questionCount === c
              return (
                <button
                  key={c}
                  onClick={() => onQuestionCountChange(c)}
                  aria-pressed={active}
                  className={`flex-1 rounded-xl py-2.5 text-sm font-semibold transition-all ${
                    active
                      ? 'bg-gradient-brand text-white shadow-[0_6px_18px_-8px_rgba(34,211,238,0.5)]'
                      : 'text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200'
                  }`}
                >
                  {c}
                </button>
              )
            })}
          </div>
        </div>

        {/* Prominent CTA */}
        <Button
          onClick={onGenerate}
          disabled={loading || !canGenerate}
          loading={loading}
          size="lg"
          leftIcon={<Timer className="h-4 w-4" />}
          className="w-full !py-3.5 !text-base"
        >
          {loading ? 'Generating your exam…' : 'Start exam'}
        </Button>
        {loading && (
          <p className="mt-3 text-center text-xs text-zinc-500">
            Retrieving from your materials and writing exam questions — this can take a moment.
          </p>
        )}

        {/* Generation failure — canonical error state with retry */}
        {genError && !loading && (
          <ErrorState
            compact
            title="Couldn't generate your exam. Check your connection and try again."
            onRetry={onGenerate}
            className="mt-4"
          />
        )}

        {/* Secondary options — past paper + sample, subtle */}
        <div className="mt-6 flex items-center justify-center gap-4 text-xs">
          <button
            onClick={() => setShowUpload((v) => !v)}
            className="text-zinc-400 transition-colors hover:text-cyan-300"
          >
            or solve a past paper
          </button>
          <span className="text-zinc-700">·</span>
          <button
            onClick={onLoadSample}
            className="text-zinc-400 transition-colors hover:text-cyan-300"
          >
            try a sample exam
          </button>
        </div>

        {/* Past-paper upload — revealed on demand, doesn't clutter the main flow */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <PastPaperPanel
                uploading={uploading}
                analysisSummary={analysisSummary}
                onUpload={onUploadPaper}
                disabled={!canGenerate}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
