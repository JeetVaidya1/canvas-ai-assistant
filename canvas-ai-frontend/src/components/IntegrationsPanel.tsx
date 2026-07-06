import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Download, Github, Upload, ChevronDown, Sparkles, GraduationCap, Users, Copy } from 'lucide-react'
import { exportCourseMarkdown, githubPush, githubImport, getContextPack, importCanvasLms, publishCourse } from '@/lib/api'
import { useUser } from '@/hooks/useUser'
import { useShareInfo } from '@/hooks/useShareInfo'
import { showError, showSuccess } from '@/lib/toast'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'

const inputClass =
  'w-full px-3 py-2 bg-surface border border-line rounded-lg text-ink placeholder-ink-faint ' +
  'hover:border-line-strong focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none text-sm transition-colors'

interface IntegrationsPanelProps {
  courseId: string
}

export default function IntegrationsPanel({ courseId }: IntegrationsPanelProps) {
  const userId = useUser()
  const [open, setOpen] = useState(false)
  const [repo, setRepo] = useState('')
  const [token, setToken] = useState('')
  const [subdir, setSubdir] = useState('')
  const [canvasUrl, setCanvasUrl] = useState('https://canvas.instructure.com')
  const [canvasToken, setCanvasToken] = useState('')
  const [canvasCourse, setCanvasCourse] = useState('')
  const [busy, setBusy] = useState<'export' | 'push' | 'import' | 'context' | 'canvas' | 'publish' | null>(null)

  const qc = useQueryClient()
  // null means "not published yet" (the hook normalizes the backend's error
  // response for unpublished courses).
  const shareInfoQuery = useShareInfo(courseId)
  const shareCode = shareInfoQuery.data?.share_code ?? null
  const joinCount = shareInfoQuery.data?.join_count ?? 0

  const handlePublish = async () => {
    setBusy('publish')
    try {
      const r = await publishCourse(courseId, userId)
      showSuccess(r.republished ? 'Catalog listing updated' : `Published — share code ${r.share_code}`)
      // Pull the fresh share info (code + join count) into the cache.
      await qc.invalidateQueries({ queryKey: ['shareInfo', courseId] })
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Publish failed')
    } finally {
      setBusy(null)
    }
  }

  const handleCanvasImport = async () => {
    if (!canvasUrl || !canvasToken || !canvasCourse) {
      showError('Canvas base URL, token, and course id are required')
      return
    }
    setBusy('canvas')
    try {
      const r = await importCanvasLms(canvasUrl, canvasToken, canvasCourse, courseId)
      if (r.next_exam_date) {
        // Hand the detected exam date to the planner (prefills the form).
        localStorage.setItem(`vindexa_exam_date_${courseId}`, r.next_exam_date)
      }
      const examNote = r.next_exam_date ? ` Next exam: ${r.next_exam_date}.` : ''
      showSuccess(`Imported ${r.materials_imported} file(s)${r.syllabus_imported ? ' + syllabus' : ''}.${examNote}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Canvas import failed')
    } finally {
      setBusy(null)
    }
  }

  const handleCopyContext = async () => {
    setBusy('context')
    try {
      const md = await getContextPack(courseId, userId)
      await navigator.clipboard.writeText(md)
      showSuccess('Study context copied — paste into Claude, ChatGPT, or a Project')
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to build context pack')
    } finally {
      setBusy(null)
    }
  }

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
    <Card padding="none">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 text-left"
      >
        <span className="text-sm font-semibold text-ink flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-lg bg-paper-deep border border-line flex items-center justify-center">
            <Github className="w-4 h-4 text-ink-soft" />
          </span>
          Export &amp; integrations
        </span>
        <ChevronDown className={`w-4 h-4 text-ink-faint transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="px-4 pb-4 space-y-4 border-t border-line pt-4">
          <div className="border-b border-line pb-4">
            {shareCode ? (
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <Users className="w-4 h-4 text-accent" />
                  <span className="text-sm font-medium text-ink">Published to the class catalog</span>
                </div>
                <div className="flex items-center gap-2">
                  <code className="px-2.5 py-1 bg-accent-wash border border-accent-line rounded text-accent-deep text-sm font-mono tracking-widest">{shareCode}</code>
                  <button
                    onClick={() => { void navigator.clipboard.writeText(shareCode); showSuccess('Share code copied') }}
                    className="p-1.5 text-ink-soft hover:text-accent hover:bg-paper-deep rounded-lg transition-colors"
                    title="Copy share code"
                    aria-label="Copy share code"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <span className="text-xs text-ink-faint">{joinCount} classmate{joinCount === 1 ? '' : 's'} joined</span>
                </div>
                <p className="text-xs text-ink-faint mt-1.5">Classmates enter this code to study the same course — each keeps their own progress.</p>
              </div>
            ) : (
              <div>
                <Button
                  onClick={() => void handlePublish()}
                  loading={busy === 'publish'}
                  leftIcon={<Users className="w-4 h-4" />}
                >
                  {busy === 'publish' ? 'Publishing…' : 'Share with your class'}
                </Button>
                <p className="text-xs text-ink-faint mt-1.5">Publish this course so classmates can join with a code. You stay the owner.</p>
              </div>
            )}
          </div>

          <div>
            <Button
              onClick={() => void handleCopyContext()}
              loading={busy === 'context'}
              leftIcon={<Sparkles className="w-4 h-4" />}
            >
              {busy === 'context' ? 'Building…' : 'Copy study context for AI'}
            </Button>
            <p className="text-xs text-ink-faint mt-1.5">A grounded brief of your weak areas + source excerpts — paste into Claude, ChatGPT, or a Project. (Vindexa also runs as an MCP server; see mcp_server.py.)</p>
          </div>

          <div>
            <Button
              variant="secondary"
              onClick={() => void handleExport()}
              loading={busy === 'export'}
              leftIcon={<Download className="w-4 h-4" />}
            >
              {busy === 'export' ? 'Exporting…' : 'Export course as Markdown (.zip)'}
            </Button>
            <p className="text-xs text-ink-faint mt-1.5">Notes, flashcards, and your study plan as version-control-friendly Markdown.</p>
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-medium text-ink-soft">GitHub repository (owner/name)</label>
            <input
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="yourname/study-notes"
              className={inputClass}
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                value={token}
                onChange={(e) => setToken(e.target.value)}
                type="password"
                placeholder="token (push / private)"
                className={inputClass}
              />
              <input
                value={subdir}
                onChange={(e) => setSubdir(e.target.value)}
                placeholder="import subdir (optional)"
                className={inputClass}
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant="secondary"
                onClick={() => void handlePush()}
                loading={busy === 'push'}
                leftIcon={<Upload className="w-4 h-4" />}
              >
                {busy === 'push' ? 'Pushing…' : 'Push to GitHub'}
              </Button>
              <Button
                variant="secondary"
                onClick={() => void handleImport()}
                loading={busy === 'import'}
                leftIcon={<Download className="w-4 h-4" />}
              >
                {busy === 'import' ? 'Importing…' : 'Import materials'}
              </Button>
            </div>
            <p className="text-xs text-ink-faint">
              Tokens are sent only with your request and never stored. Importing pulls text/Markdown files into this course.
            </p>
          </div>

          <div className="space-y-2 border-t border-line pt-4">
            <label className="block text-xs font-medium text-ink-soft flex items-center gap-1.5">
              <GraduationCap className="w-3.5 h-3.5 text-accent" /> Import from Canvas LMS
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                value={canvasUrl}
                onChange={(e) => setCanvasUrl(e.target.value)}
                placeholder="https://canvas.school.edu"
                className={inputClass}
              />
              <input
                value={canvasCourse}
                onChange={(e) => setCanvasCourse(e.target.value)}
                placeholder="Canvas course id"
                className={inputClass}
              />
            </div>
            <input
              value={canvasToken}
              onChange={(e) => setCanvasToken(e.target.value)}
              type="password"
              placeholder="Canvas access token"
              className={inputClass}
            />
            <Button
              variant="secondary"
              onClick={() => void handleCanvasImport()}
              loading={busy === 'canvas'}
              leftIcon={<GraduationCap className="w-4 h-4" />}
            >
              {busy === 'canvas' ? 'Importing…' : 'Import syllabus, materials & exam dates'}
            </Button>
            <p className="text-xs text-ink-faint">Pulls your syllabus and files, and detects your next exam to prefill the planner.</p>
          </div>
        </div>
      )}
    </Card>
  )
}
