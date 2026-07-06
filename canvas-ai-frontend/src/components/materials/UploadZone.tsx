import { useRef, useState } from 'react'
import { Upload, FileText, X } from 'lucide-react'
import { useUploadFile } from '@/hooks/useCourseFiles'
import { Button } from '@/components/ui/Button'
import { ErrorState } from '@/components/ui/States'
import { cn } from '@/lib/utils'

const ACCEPT = '.pdf,.docx,.pptx,.txt,.md'

interface UploadZoneProps {
  courseId: string
  /** Larger hero treatment for the empty-course onboarding state. */
  prominent?: boolean
}

/**
 * Drag-and-drop upload with an honest in-flight state: upload + indexing is a
 * single backend await, so we show a staged indeterminate treatment — never a
 * fake percentage.
 */
export default function UploadZone({ courseId, prominent = false }: UploadZoneProps) {
  const uploadFile = useUploadFile(courseId)
  const inputRef = useRef<HTMLInputElement>(null)
  const [pending, setPending] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [failed, setFailed] = useState(false)

  const addFiles = (list: FileList | null) => {
    if (!list) return
    setFailed(false)
    setPending((prev) => {
      const names = new Set(prev.map((f) => f.name))
      return [...prev, ...Array.from(list).filter((f) => !names.has(f.name))]
    })
  }

  const removePending = (name: string) => setPending((prev) => prev.filter((f) => f.name !== name))

  const handleUpload = async () => {
    if (pending.length === 0) return
    setFailed(false)
    try {
      await uploadFile.mutateAsync(pending)
      setPending([])
    } catch {
      setFailed(true)
    }
  }

  const busy = uploadFile.isPending

  return (
    <div className="space-y-4">
      <label
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={(e) => { e.preventDefault(); setDragActive(false) }}
        onDrop={(e) => {
          e.preventDefault()
          setDragActive(false)
          addFiles(e.dataTransfer.files)
        }}
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed cursor-pointer transition-all focus-ring',
          prominent ? 'px-6 py-10' : 'px-5 py-7',
          dragActive
            ? 'border-cyan-400/60 bg-gradient-brand-soft'
            : 'border-border-strong bg-white/[0.02] hover:border-cyan-400/40 hover:bg-white/[0.04]',
          busy && 'pointer-events-none opacity-60',
        )}
      >
        <Upload className={cn('w-6 h-6 transition-colors', dragActive ? 'text-cyan-300' : 'text-zinc-500')} />
        <span className="text-sm font-medium text-zinc-300">Drop files here or click to browse</span>
        <span className="text-xs text-zinc-600">PDF, DOCX, PPTX, TXT, MD</span>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          onChange={(e) => {
            addFiles(e.target.files)
            e.target.value = ''
          }}
          className="hidden"
          disabled={busy}
        />
      </label>

      {pending.length > 0 && (
        <div className="rounded-xl border border-border bg-bg-subtle/60 p-4 space-y-3">
          <ul className="space-y-1.5">
            {pending.map((f) => (
              <li key={f.name} className="flex items-center gap-2.5 text-sm text-zinc-300">
                <FileText className="w-4 h-4 text-cyan-300/70 flex-shrink-0" />
                <span className="truncate flex-1">{f.name}</span>
                {!busy && (
                  <button
                    onClick={() => removePending(f.name)}
                    className="p-1 text-zinc-600 hover:text-zinc-300 rounded transition-colors"
                    aria-label={`Remove ${f.name}`}
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </li>
            ))}
          </ul>

          {busy ? (
            <div className="space-y-2">
              <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
                <div className="h-full w-1/3 rounded-full bg-gradient-brand animate-sheen" />
              </div>
              <p className="text-xs text-zinc-500">
                Uploading and indexing — chunking, embedding and linking your {pending.length === 1 ? 'file' : 'files'} into the course knowledge base…
              </p>
            </div>
          ) : (
            <Button onClick={handleUpload} className="w-full justify-center" leftIcon={<Upload className="w-4 h-4" />}>
              Upload {pending.length} {pending.length === 1 ? 'file' : 'files'}
            </Button>
          )}
        </div>
      )}

      {failed && (
        <ErrorState
          compact
          title="Upload failed — your files are still listed above."
          onRetry={handleUpload}
          retrying={busy}
        />
      )}
    </div>
  )
}
