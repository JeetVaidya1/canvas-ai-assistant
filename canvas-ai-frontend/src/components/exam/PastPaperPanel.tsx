// Opt-in past-paper upload section on the exam setup screen.
import { useRef } from 'react'
import { CheckCircle, FileText, Upload } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { EmptyState } from '@/components/ui/States'
import type { PastPaperAnalysis } from './types'

interface PastPaperPanelProps {
  uploading: boolean
  analysisSummary: PastPaperAnalysis | null
  onUpload: (file: File) => void
  disabled: boolean
}

/**
 * Upload a past exam PDF so the backend can analyze it and generate similar
 * questions. Shows an EmptyState until the first paper is uploaded, then the
 * analysis outcome as a semantic Badge.
 */
export function PastPaperPanel({ uploading, analysisSummary, onUpload, disabled }: PastPaperPanelProps) {
  const fileRef = useRef<HTMLInputElement>(null)

  const pickFile = () => fileRef.current?.click()

  return (
    <div className="mt-5 card-surface rounded-xl">
      <input
        ref={fileRef}
        type="file"
        accept=".pdf"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onUpload(f)
          e.target.value = ''
        }}
      />

      {!analysisSummary ? (
        <EmptyState
          icon={<FileText />}
          title="No past papers yet"
          description="Upload a past exam PDF and we'll analyze its style to generate similar practice questions."
          action={
            <Button
              variant="secondary"
              size="sm"
              onClick={pickFile}
              disabled={disabled}
              loading={uploading}
              leftIcon={<Upload className="w-3.5 h-3.5" />}
            >
              {uploading ? 'Uploading…' : 'Choose PDF'}
            </Button>
          }
          className="py-8"
        />
      ) : (
        <div className="p-5 text-center">
          <h3 className="text-sm font-semibold text-ink mb-2.5">Past paper analysis</h3>
          {analysisSummary.status === 'success' ? (
            <Badge tone="success" icon={<CheckCircle />}>
              {analysisSummary.questions_found ?? 0} questions found
            </Badge>
          ) : (
            <Badge tone="danger">{analysisSummary.message ?? 'Upload failed'}</Badge>
          )}
          <div className="mt-3.5">
            <Button
              variant="ghost"
              size="sm"
              onClick={pickFile}
              disabled={disabled}
              loading={uploading}
              leftIcon={<Upload className="w-3.5 h-3.5" />}
            >
              {uploading ? 'Uploading…' : 'Upload another'}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
