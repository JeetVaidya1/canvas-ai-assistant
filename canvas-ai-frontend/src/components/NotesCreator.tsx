// src/components/NotesCreator.tsx — thin composition over the Study Kit modules in ./studykit/
// State + server interactions live in useNotesStudio; each surface is its own component.
import { Bookmark, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import NotesComposer from './studykit/NotesComposer'
import NoteView from './studykit/NoteView'
import NotesLibrary from './studykit/NotesLibrary'
import { GenerationProgress } from './studykit/GenerationProgress'
import { GeneratedFlashcardsPanel, SpacedRepetitionPanel } from './studykit/GeneratedFlashcards'
import { useNotesStudio } from './studykit/useNotesStudio'

interface NotesCreatorProps {
  courseId: string
  courseName: string
}

export default function NotesCreator({ courseId }: NotesCreatorProps) {
  const studio = useNotesStudio(courseId)
  const {
    userId, savedNotes, selectedFiles, availableFiles, usingAllFiles,
    topic, generatedNotes, flashcards, loading, loadStage,
    setLibraryOpen, reset,
  } = studio

  const showReader = !loading && !!generatedNotes
  const dirty = !!generatedNotes || !usingAllFiles || !!topic
  const generatingFileCount = selectedFiles.length > 0 ? selectedFiles.length : availableFiles.length

  return (
    <div className="relative flex min-h-full flex-col">
      {/* Floating controls — mirrors ChatPage's History button */}
      <div className="absolute right-0 top-0 z-20 flex items-center gap-2">
        {dirty && (
          <Button
            variant="secondary"
            size="sm"
            onClick={reset}
            leftIcon={<RotateCcw className="h-3.5 w-3.5" />}
            className="backdrop-blur"
          >
            <span className="hidden sm:inline">New</span>
          </Button>
        )}
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setLibraryOpen(true)}
          leftIcon={<Bookmark className="h-3.5 w-3.5" />}
          className="backdrop-blur"
        >
          <span className="hidden sm:inline">Library</span>
          {savedNotes.length > 0 && (
            <span className="rounded-full bg-white/[0.1] px-1.5 text-[11px] text-zinc-400">
              {savedNotes.length}
            </span>
          )}
        </Button>
      </div>

      {/* Main content: centered studio, honest staged progress, or the reader */}
      {loading ? (
        <div className="flex-1 pt-14">
          <GenerationProgress stage={loadStage} fileCount={generatingFileCount} />
        </div>
      ) : showReader ? (
        <div className="flex-1 pt-14">
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
            <NoteView studio={studio} className="xl:col-span-2" />
            <div className="space-y-6 xl:col-span-1">
              <GeneratedFlashcardsPanel cards={flashcards} />
            </div>
            {flashcards.length > 0 && (
              <div className="xl:col-span-3">
                <SpacedRepetitionPanel cards={flashcards} courseId={courseId} userId={userId} />
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1">
          <NotesComposer key={courseId} studio={studio} />
        </div>
      )}

      <NotesLibrary key={`library-${courseId}`} studio={studio} />
    </div>
  )
}
