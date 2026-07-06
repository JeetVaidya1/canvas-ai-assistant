// src/components/studykit/NotesComposer.tsx — center-first studio: topic prompt + style pills + collapsible sources
import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronDown, FileText, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { ErrorState } from '@/components/ui/States'
import { BrandMark } from '@/components/ui/BrandMark'
import ErrorInline from '@/components/shared/ErrorInline'
import { NOTE_STYLES } from './noteUtils'
import { SourceCheckbox } from './SourceCheckbox'
import type { NotesStudio } from './useNotesStudio'

/**
 * The approved center-first interaction: a chat-composer-style topic input,
 * inline style pills, and a single collapsible source picker that defaults to
 * every file in the course.
 */
export default function NotesComposer({ studio }: { studio: NotesStudio }) {
  const {
    courseId, filesQuery, availableFiles, selectedFiles,
    allSelected, usingAllFiles, toggleFile, toggleSelectAll,
    topic, setTopic, noteStyle, setNoteStyle,
    loading, errMsg, generate,
  } = studio

  const [sourcesOpen, setSourcesOpen] = useState(false)
  const noFiles = availableFiles.length === 0

  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-2xl"
      >
        {/* Heading */}
        <div className="mb-6 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand" />
          <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
            Create study notes
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
            Grounded in your materials, with auto-generated flashcards.
          </p>
        </div>

        {/* Centerpiece: focus-topic input (styled like the chat composer) */}
        <div
          className={cn(
            'relative flex w-full items-center rounded-[20px] border border-white/12 bg-white/[0.03] p-2 shadow-lg transition-all',
            'focus-within:border-cyan-400/60 focus-within:bg-white/[0.05] focus-within:glow-brand-sm',
          )}
        >
          <span className="pl-3 pr-1 text-cyan-300/70">
            <Sparkles className="h-5 w-5" />
          </span>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !noFiles && !loading) void generate()
            }}
            placeholder="What should these notes cover?  (e.g. Binary Search Trees — or leave blank for the whole course)"
            className="flex-1 bg-transparent px-2 py-2.5 text-[15px] text-zinc-100 placeholder-zinc-500 outline-none"
          />
        </div>

        {/* Style — inline segmented pills */}
        <div className="mt-4 flex items-center justify-center gap-2">
          {NOTE_STYLES.map((style) => {
            const active = noteStyle === style.value
            return (
              <button
                key={style.value}
                type="button"
                onClick={() => setNoteStyle(style.value)}
                title={style.hint}
                className={cn(
                  'rounded-full border px-4 py-1.5 text-[13px] font-medium transition-all',
                  active
                    ? 'bg-gradient-brand-soft border-cyan-400/40 text-cyan-100 ring-1 ring-inset ring-cyan-400/30'
                    : 'border-white/10 bg-white/[0.02] text-zinc-300 hover:border-white/20 hover:bg-white/[0.05] hover:text-zinc-100',
                )}
              >
                {style.label}
              </button>
            )
          })}
        </div>

        {/* Sources — single collapsible control, defaults to ALL selected */}
        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={() => setSourcesOpen((v) => !v)}
            disabled={noFiles || filesQuery.isLoading}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-zinc-300 transition-colors hover:border-cyan-400/40 hover:bg-white/[0.06] hover:text-zinc-100 disabled:opacity-50"
          >
            <FileText className="h-3.5 w-3.5 text-cyan-300/80" />
            {filesQuery.isLoading
              ? 'Loading your files…'
              : noFiles
                ? 'No files in this course'
                : usingAllFiles
                  ? `Using all ${availableFiles.length} file${availableFiles.length === 1 ? '' : 's'}`
                  : `Using ${selectedFiles.length} of ${availableFiles.length} files`}
            {!noFiles && !filesQuery.isLoading && (
              <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', sourcesOpen && 'rotate-180')} />
            )}
          </button>
        </div>

        <AnimatePresence initial={false}>
          {sourcesOpen && !noFiles && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <Card padding="md" className="mt-3">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-medium text-zinc-400">Pick the files to ground these notes in</span>
                  <button
                    type="button"
                    onClick={toggleSelectAll}
                    className="text-xs font-medium text-cyan-300 transition-colors hover:text-cyan-200"
                  >
                    {allSelected ? 'Clear' : 'Select all'}
                  </button>
                </div>
                <div className="grid max-h-72 grid-cols-1 gap-1.5 overflow-auto pr-1 sm:grid-cols-2">
                  {availableFiles.map((file) => {
                    const checked = selectedFiles.includes(file)
                    return (
                      <button
                        type="button"
                        key={file}
                        onClick={() => toggleFile(file)}
                        title={file}
                        aria-pressed={checked}
                        className={cn(
                          'group flex items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-all',
                          checked
                            ? 'border-cyan-400/40 bg-gradient-brand-soft'
                            : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]',
                        )}
                      >
                        <SourceCheckbox checked={checked} />
                        <span className={cn('truncate text-sm', checked ? 'font-medium text-zinc-100' : 'text-zinc-300')}>
                          {file}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Primary action */}
        <Button
          size="lg"
          onClick={() => void generate()}
          loading={loading}
          disabled={loading || noFiles || !courseId}
          leftIcon={<Sparkles className="w-5 h-5" />}
          className="mt-5 w-full"
        >
          {loading ? 'Generating…' : 'Generate notes'}
        </Button>

        {filesQuery.isError && (
          <ErrorInline
            message="Couldn't load your course files."
            onRetry={() => void filesQuery.refetch()}
            className="mt-3"
          />
        )}
        {noFiles && courseId && !filesQuery.isError && !filesQuery.isLoading && (
          <p className="mt-2.5 text-center text-xs text-zinc-500">
            Upload course files from Materials to generate notes.
          </p>
        )}
        {!courseId && (
          <p className="mt-2.5 text-center text-xs text-amber-400">Select a course to get started.</p>
        )}
        {errMsg && (
          <ErrorState compact title={errMsg} onRetry={() => void generate()} className="mt-3" />
        )}
      </motion.div>
    </div>
  )
}
