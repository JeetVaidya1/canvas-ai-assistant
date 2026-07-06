// src/components/studykit/NoteView.tsx — rendered note with inline title/content editing
import { useState } from 'react'
import { BookOpen, Clock, Copy, Edit3, Eye, FileText, Save } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Markdown } from '@/components/ui/Markdown'
import { ErrorState } from '@/components/ui/States'
import { getReadingTime, getWordCount } from './noteUtils'
import type { NotesStudio } from './useNotesStudio'

const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text || '')
  } catch {
    // non-blocking
  }
}

/**
 * The generated (or reopened) note: editable title, Markdown preview with a
 * raw-edit toggle, copy, and save/update through the notes mutation hook.
 * Paper & Ink: the note reads like a typeset page — serif title, prose set
 * directly on the white sheet at a comfortable measure.
 */
export default function NoteView({ studio, className }: { studio: NotesStudio; className?: string }) {
  const {
    generatedNotes, setGeneratedNotes,
    noteTitle, setNoteTitle,
    currentNoteId, saving, save, errMsg, selectedFiles,
  } = studio
  const [previewMode, setPreviewMode] = useState(true)

  return (
    <Card padding="none" className={`overflow-hidden ${className ?? ''}`}>
      <div className="px-6 py-4 border-b border-line">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <Edit3 className="w-5 h-5 text-accent flex-shrink-0" />
            <input
              type="text"
              value={noteTitle}
              onChange={(e) => setNoteTitle(e.target.value)}
              placeholder="Enter note title…"
              className="font-display text-xl font-semibold bg-transparent border-none focus:outline-none focus:ring-0 text-ink placeholder-ink-faint min-w-0 w-full"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPreviewMode(!previewMode)}
              title={previewMode ? 'Edit mode' : 'Preview mode'}
              aria-label={previewMode ? 'Switch to edit mode' : 'Switch to preview mode'}
              className="!px-2"
            >
              {previewMode ? <Edit3 className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void copyToClipboard(generatedNotes)}
              title="Copy notes"
              aria-label="Copy notes"
              className="!px-2"
            >
              <Copy className="w-4 h-4" />
            </Button>
            <Button
              onClick={() => void save()}
              loading={saving}
              disabled={saving || !generatedNotes.trim() || !noteTitle.trim()}
              leftIcon={<Save className="w-4 h-4" />}
            >
              {saving
                ? (currentNoteId ? 'Updating…' : 'Saving…')
                : (currentNoteId ? 'Update' : 'Save')}
            </Button>
          </div>
        </div>
      </div>

      <div className="px-6 py-7 sm:px-10">
        {errMsg && (
          <ErrorState compact title={errMsg} onRetry={() => void save()} retrying={saving} className="mb-4" />
        )}

        {previewMode ? (
          <div className="mx-auto max-w-[70ch]">
            <Markdown content={generatedNotes} />
          </div>
        ) : (
          <textarea
            value={generatedNotes}
            onChange={(e) => setGeneratedNotes(e.target.value)}
            className="w-full h-[28rem] p-4 bg-paper-deep border border-line text-ink rounded-lg outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 resize-none font-mono text-sm transition-colors"
            placeholder="Your generated notes will appear here…"
          />
        )}

        {generatedNotes && (
          <div className="mt-6 pt-4 border-t border-line flex items-center gap-6 text-sm text-ink-soft flex-wrap">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>{getReadingTime(getWordCount(generatedNotes))}</span>
            </div>
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              <span>{getWordCount(generatedNotes)} words</span>
            </div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              <span>{selectedFiles.length} source file(s)</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
