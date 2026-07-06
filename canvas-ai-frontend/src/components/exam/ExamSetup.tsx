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
          <BrandMark className="mx-auto mb-5 h-14 w-14" />
          <h1 className="font-display text-[28px] font-semibold tracking-tight text-ink">
            Set up your mock exam
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-ink-soft">
            A timed simulation built from your materials. Answers and the clock auto-save — close the tab and pick up
            exactly where you left off. AI judges with partial credit and grounded explanations.
          </p>
        </div>

        {/* Difficulty — 4 big selectable tiles */}
        <div className="mb-6">
          <div className="mb-2.5 text-center text-xs font-medium text-ink-soft">Difficulty</div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {DIFFICULTIES.map((d) => {
              const active = difficulty === d.value
              return (
                <button
                  key={d.value}
                  onClick={() => onDifficultyChange(d.value)}
                  aria-pressed={active}
                  className={`group rounded-xl border px-3 py-4 text-center transition-all ${
                    active
                      ? 'border-accent bg-accent-wash ring-2 ring-accent/20'
                      : 'card-surface card-interactive'
                  }`}
                >
                  <div className={`text-sm font-semibold ${active ? 'text-accent-deep' : 'text-ink'}`}>
                    {d.label}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-faint leading-tight">{d.hint}</div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Question count — segmented control */}
        <div className="mb-8">
          <div className="mb-2.5 text-center text-xs font-medium text-ink-soft">Questions</div>
          <div className="flex gap-1.5 rounded-xl border border-line bg-paper-deep p-1.5">
            {COUNTS.map((c) => {
              const active = questionCount === c
              return (
                <button
                  key={c}
                  onClick={() => onQuestionCountChange(c)}
                  aria-pressed={active}
                  className={`flex-1 rounded-lg py-2.5 text-sm font-semibold tnum transition-all ${
                    active
                      ? 'bg-accent text-white elev-1'
                      : 'text-ink-soft hover:bg-surface hover:text-ink'
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
          <p className="mt-3 text-center text-xs text-ink-faint">
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
            className="text-ink-soft transition-colors hover:text-accent"
          >
            or solve a past paper
          </button>
          <span className="text-ink-faint">·</span>
          <button
            onClick={onLoadSample}
            className="text-ink-soft transition-colors hover:text-accent"
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
