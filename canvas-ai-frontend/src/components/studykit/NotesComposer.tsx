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
          <BrandMark className="mx-auto mb-5 h-14 w-14" />
          <h1 className="font-display text-[28px] font-semibold tracking-tight text-ink">
            Create study notes
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-ink-soft">
            Grounded in your materials, with auto-generated flashcards.
          </p>
        </div>

        {/* Centerpiece: focus-topic input (styled like the chat composer) */}
        <div
          className={cn(
            'relative flex w-full items-center rounded-[20px] border border-line bg-surface p-2 elev-1 transition-all',
            'focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20',
          )}
        >
          <span className="pl-3 pr-1 text-accent">
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
            className="flex-1 bg-transparent px-2 py-2.5 text-[15px] text-ink placeholder-ink-faint outline-none"
          />
        </div>

        {/* Style — inline hairline pills */}
        <div className="mt-4 flex items-center justify-center gap-2">
          {NOTE_STYLES.map((style) => {
            const active = noteStyle === style.value
            return (
              <button
                key={style.value}
                type="button"
                onClick={() => setNoteStyle(style.value)}
                title={style.hint}
                aria-pressed={active}
                className={cn(
                  'rounded-full border px-4 py-1.5 text-[13px] font-medium transition-all focus-ring',
                  active
                    ? 'border-accent-line bg-accent-wash text-accent-deep'
                    : 'border-line bg-surface text-ink-soft hover:border-line-strong hover:text-ink',
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
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3.5 py-1.5 text-[13px] text-ink-soft transition-colors hover:border-line-strong hover:bg-surface-hover hover:text-ink disabled:opacity-50 focus-ring"
          >
            <FileText className="h-3.5 w-3.5 text-ink-faint" />
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
                  <span className="text-xs font-medium text-ink-soft">Pick the files to ground these notes in</span>
                  <button
                    type="button"
                    onClick={toggleSelectAll}
                    className="text-xs font-medium text-accent transition-colors hover:text-accent-deep"
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
                            ? 'border-accent-line bg-accent-wash'
                            : 'border-line bg-surface hover:border-line-strong hover:bg-surface-hover',
                        )}
                      >
                        <SourceCheckbox checked={checked} />
                        <span className={cn('truncate text-sm', checked ? 'font-medium text-ink' : 'text-ink-soft')}>
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
          <p className="mt-2.5 text-center text-xs text-ink-faint">
            Upload course files from Materials to generate notes.
          </p>
        )}
        {!courseId && (
          <p className="mt-2.5 text-center text-xs text-warning">Select a course to get started.</p>
        )}
        {errMsg && (
          <ErrorState compact title={errMsg} onRetry={() => void generate()} className="mt-3" />
        )}
      </motion.div>
    </div>
  )
}
