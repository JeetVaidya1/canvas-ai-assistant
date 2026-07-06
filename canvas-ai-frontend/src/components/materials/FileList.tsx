import { useState } from 'react'
import { FileText, Trash2, FolderOpen } from 'lucide-react'
import { useCourseFiles, useDeleteFile } from '@/hooks/useCourseFiles'
import ConfirmDialog from '@/components/shared/ConfirmDialog'
import { EmptyState, ErrorState } from '@/components/ui/States'

interface FileListProps {
  courseId: string
}

/** Indexed-files list with skeleton / error / empty coverage and confirmed deletes. */
export default function FileList({ courseId }: FileListProps) {
  const filesQuery = useCourseFiles(courseId)
  const deleteFile = useDeleteFile(courseId)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)

  if (filesQuery.isPending) {
    return (
      <div className="space-y-2" aria-label="Loading files">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-10 rounded-lg bg-paper-deep animate-pulse" />
        ))}
      </div>
    )
  }

  if (filesQuery.isError) {
    return (
      <ErrorState
        compact
        title="Couldn’t load this course’s files."
        onRetry={() => void filesQuery.refetch()}
        retrying={filesQuery.isRefetching}
      />
    )
  }

  const files = filesQuery.data ?? []
  if (files.length === 0) {
    return (
      <EmptyState
        icon={<FolderOpen />}
        title="No materials indexed yet"
        description="Everything Vindexa generates is grounded in these files — add lecture slides, readings, and past papers above."
        className="py-8"
      />
    )
  }

  return (
    <>
      <ul className="space-y-1">
        {files.map((filename) => (
          <li
            key={filename}
            className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-paper-deep/60 transition-colors group"
          >
            <span className="text-ink text-sm truncate flex-1 flex items-center gap-2.5">
              <FileText className="w-4 h-4 text-ink-faint flex-shrink-0" />
              {filename}
            </span>
            <button
              onClick={() => setDeleteTarget(filename)}
              className="p-1.5 text-ink-faint hover:text-danger hover:bg-danger-wash rounded-lg transition-all opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
              aria-label={`Delete ${filename}`}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </li>
        ))}
      </ul>

      <ConfirmDialog
        open={!!deleteTarget}
        title="Remove file"
        description={`Remove "${deleteTarget}" from this course? Its content will no longer ground answers, quizzes, or notes.`}
        confirmLabel="Remove"
        variant="danger"
        onConfirm={() => {
          if (deleteTarget) deleteFile.mutate(deleteTarget)
          setDeleteTarget(null)
        }}
        onCancel={() => setDeleteTarget(null)}
      />
    </>
  )
}
