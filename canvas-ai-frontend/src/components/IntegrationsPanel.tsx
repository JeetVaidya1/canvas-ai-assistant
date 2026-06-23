import { useState, useEffect } from 'react'
import { Download, Github, Upload, ChevronDown, Sparkles, GraduationCap, Users, Copy } from 'lucide-react'
import { exportCourseMarkdown, githubPush, githubImport, getContextPack, importCanvasLms, publishCourse, getShareInfo } from '@/lib/api'
import { useUser } from '@/hooks/useUser'
import { showError, showSuccess } from '@/lib/toast'

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
  const [shareCode, setShareCode] = useState<string | null>(null)
  const [joinCount, setJoinCount] = useState(0)

  useEffect(() => {
    if (!courseId) return
    getShareInfo(courseId).then((info) => {
      if (info) { setShareCode(info.share_code); setJoinCount(info.join_count || 0) }
    }).catch(() => { /* not published yet */ })
  }, [courseId])

  const handlePublish = async () => {
    setBusy('publish')
    try {
      const r = await publishCourse(courseId, userId)
      setShareCode(r.share_code)
      showSuccess(r.republished ? 'Catalog listing updated' : `Published — share code ${r.share_code}`)
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
          <div className="border-b border-zinc-700/40 pb-4">
            {shareCode ? (
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <Users className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm font-medium text-zinc-100">Published to the class catalog</span>
                </div>
                <div className="flex items-center gap-2">
                  <code className="px-2.5 py-1 bg-zinc-900 border border-zinc-700 rounded text-emerald-400 text-sm font-mono tracking-widest">{shareCode}</code>
                  <button
                    onClick={() => { void navigator.clipboard.writeText(shareCode); showSuccess('Share code copied') }}
                    className="p-1.5 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg"
                    title="Copy share code"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                  <span className="text-xs text-zinc-500">{joinCount} classmate{joinCount === 1 ? '' : 's'} joined</span>
                </div>
                <p className="text-xs text-zinc-500 mt-1.5">Classmates enter this code to study the same course — each keeps their own progress.</p>
              </div>
            ) : (
              <div>
                <button
                  onClick={() => void handlePublish()}
                  disabled={busy === 'publish'}
                  className="bg-emerald-600 text-white px-3 py-2 rounded-lg hover:bg-emerald-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
                >
                  <Users className="w-4 h-4" /> {busy === 'publish' ? 'Publishing…' : 'Share with your class'}
                </button>
                <p className="text-xs text-zinc-500 mt-1.5">Publish this course so classmates can join with a code. You stay the owner.</p>
              </div>
            )}
          </div>

          <div>
            <button
              onClick={() => void handleCopyContext()}
              disabled={busy === 'context'}
              className="bg-cyan-600 text-white px-3 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              {busy === 'context' ? 'Building…' : 'Copy study context for AI'}
            </button>
            <p className="text-xs text-zinc-500 mt-1.5">A grounded brief of your weak areas + source excerpts — paste into Claude, ChatGPT, or a Project. (Vindexa also runs as an MCP server; see mcp_server.py.)</p>
          </div>

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

          <div className="space-y-2 border-t border-zinc-700/40 pt-4">
            <label className="block text-xs font-medium text-zinc-500 flex items-center gap-1.5">
              <GraduationCap className="w-3.5 h-3.5" /> Import from Canvas LMS
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                value={canvasUrl}
                onChange={(e) => setCanvasUrl(e.target.value)}
                placeholder="https://canvas.school.edu"
                className="px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
              />
              <input
                value={canvasCourse}
                onChange={(e) => setCanvasCourse(e.target.value)}
                placeholder="Canvas course id"
                className="px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
              />
            </div>
            <input
              value={canvasToken}
              onChange={(e) => setCanvasToken(e.target.value)}
              type="password"
              placeholder="Canvas access token"
              className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
            <button
              onClick={() => void handleCanvasImport()}
              disabled={busy === 'canvas'}
              className="bg-zinc-800 border border-zinc-700 text-zinc-200 px-3 py-2 rounded-lg hover:bg-zinc-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
            >
              <GraduationCap className="w-4 h-4" /> {busy === 'canvas' ? 'Importing…' : 'Import syllabus, materials & exam dates'}
            </button>
            <p className="text-xs text-zinc-500">Pulls your syllabus and files, and detects your next exam to prefill the planner.</p>
          </div>
        </div>
      )}
    </div>
  )
}
