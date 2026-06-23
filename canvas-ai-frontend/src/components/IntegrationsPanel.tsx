import { useState } from 'react'
import { Download, Github, Upload, ChevronDown } from 'lucide-react'
import { exportCourseMarkdown, githubPush, githubImport } from '@/lib/api'
import { showError, showSuccess } from '@/lib/toast'

interface IntegrationsPanelProps {
  courseId: string
}

export default function IntegrationsPanel({ courseId }: IntegrationsPanelProps) {
  const [open, setOpen] = useState(false)
  const [repo, setRepo] = useState('')
  const [token, setToken] = useState('')
  const [subdir, setSubdir] = useState('')
  const [busy, setBusy] = useState<'export' | 'push' | 'import' | null>(null)

  const handleExport = async () => {
    setBusy('export')
    try {
      const blob = await exportCourseMarkdown(courseId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${courseId}_markdown.zip`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setBusy(null)
    }
  }

  const handlePush = async () => {
    if (!repo || !token) {
      showError('Repo (owner/name) and a token with write access are required')
      return
    }
    setBusy('push')
    try {
      const r = await githubPush(courseId, repo, token)
      showSuccess(`Pushed ${r.pushed} file(s) to ${r.repo}@${r.branch}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Push failed')
    } finally {
      setBusy(null)
    }
  }

  const handleImport = async () => {
    if (!repo) {
      showError('Repo (owner/name) is required')
      return
    }
    setBusy('import')
    try {
      const r = await githubImport(courseId, repo, token || undefined, subdir)
      showSuccess(r.message || `Imported ${r.imported} file(s), skipped ${r.skipped}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 text-left"
      >
        <span className="text-sm font-medium text-zinc-100 flex items-center gap-2">
          <Github className="w-4 h-4 text-zinc-400" /> Export &amp; integrations
        </span>
        <ChevronDown className={`w-4 h-4 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4 border-t border-zinc-700/40 pt-4">
          <div>
            <button
              onClick={() => void handleExport()}
              disabled={busy === 'export'}
              className="bg-zinc-800 border border-zinc-700 text-zinc-200 px-3 py-2 rounded-lg hover:bg-zinc-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              {busy === 'export' ? 'Exporting…' : 'Export course as Markdown (.zip)'}
            </button>
            <p className="text-xs text-zinc-500 mt-1.5">Notes, flashcards, and your study plan as version-control-friendly Markdown.</p>
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-medium text-zinc-500">GitHub repository (owner/name)</label>
            <input
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="yourname/study-notes"
              className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                value={token}
                onChange={(e) => setToken(e.target.value)}
                type="password"
                placeholder="token (push / private)"
                className="px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
              />
              <input
                value={subdir}
                onChange={(e) => setSubdir(e.target.value)}
                placeholder="import subdir (optional)"
                className="px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => void handlePush()}
                disabled={busy === 'push'}
                className="bg-zinc-800 border border-zinc-700 text-zinc-200 px-3 py-2 rounded-lg hover:bg-zinc-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
              >
                <Upload className="w-4 h-4" /> {busy === 'push' ? 'Pushing…' : 'Push to GitHub'}
              </button>
              <button
                onClick={() => void handleImport()}
                disabled={busy === 'import'}
                className="bg-zinc-800 border border-zinc-700 text-zinc-200 px-3 py-2 rounded-lg hover:bg-zinc-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
              >
                <Download className="w-4 h-4" /> {busy === 'import' ? 'Importing…' : 'Import materials'}
              </button>
            </div>
            <p className="text-xs text-zinc-500">
              Tokens are sent only with your request and never stored. Importing pulls text/Markdown files into this course.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
