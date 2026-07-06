import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Download, User, LogOut, Link2 } from 'lucide-react'
import { useProfile } from '@/hooks/useProfile'
import { useAuth } from '@/lib/auth'
import { useCourses } from '@/hooks/useCourses'
import { exportNotesPdf } from '@/lib/api/notes'
import { exportFlashcardsAnki } from '@/lib/api/flashcards'
import { exportPlannerIcal } from '@/lib/api/planner'
import { showSuccess, showError } from '@/lib/toast'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'

const inputClass =
  'w-full px-3 py-2 bg-surface border border-line rounded-lg text-ink placeholder-ink-faint ' +
  'hover:border-line-strong focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none text-sm transition-colors'

const sectionIconClass =
  'w-10 h-10 rounded-xl bg-paper-deep border border-line flex items-center justify-center flex-shrink-0'

type ExportKind = 'notes' | 'anki' | 'ical'

const EXPORTS: ReadonlyArray<{ kind: ExportKind; label: string; filename: (courseId: string) => string }> = [
  { kind: 'notes', label: 'Notes (PDF)', filename: (c) => `${c}-notes.pdf` },
  { kind: 'anki', label: 'Flashcards (Anki)', filename: (c) => `${c}-flashcards.apkg` },
  { kind: 'ical', label: 'Study plan (iCal)', filename: (c) => `${c}-plan.ics` },
]

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function SettingsPage() {
  const { displayName, setDisplayName } = useProfile()
  const { user, signOut } = useAuth()
  const { data: courses } = useCourses()
  const navigate = useNavigate()
  const [nameInput, setNameInput] = useState(displayName)
  const [exportCourse, setExportCourse] = useState('')
  const [exporting, setExporting] = useState<ExportKind | null>(null)

  const handleSaveProfile = () => {
    setDisplayName(nameInput.trim())
    showSuccess('Display name updated')
  }

  const handleExport = async (kind: ExportKind) => {
    if (!exportCourse) {
      showError('Pick a course to export from')
      return
    }
    setExporting(kind)
    try {
      const spec = EXPORTS.find((e) => e.kind === kind)
      if (!spec) return
      const blob =
        kind === 'notes' ? await exportNotesPdf(exportCourse)
        : kind === 'anki' ? await exportFlashcardsAnki(exportCourse)
        : await exportPlannerIcal(exportCourse)
      downloadBlob(blob, spec.filename(exportCourse))
      showSuccess('Export ready')
    } catch {
      showError('Export failed — make sure this course has content to export')
    } finally {
      setExporting(null)
    }
  }

  const handleSignOut = async () => {
    await signOut()
    navigate('/login')
  }

  const initial = (displayName || user?.email || 'U').charAt(0).toUpperCase()
  const courseOptions = (courses ?? []).map((c) => ({ value: c.course_id, label: c.title }))

  return (
    <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
      <PageHeader eyebrow="Settings" title="Settings" />

      {/* Account */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <User className="w-5 h-5 text-ink-soft" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-ink">Account</h2>
            {user?.email && <p className="text-xs text-ink-faint mt-0.5">{user.email}</p>}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-ink flex items-center justify-center font-display text-lg font-semibold text-paper flex-shrink-0">
            {initial}
          </div>
          <div className="flex-1 space-y-2">
            <input
              type="text"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              placeholder="Display name"
              className={inputClass}
            />
            <Button
              onClick={handleSaveProfile}
              disabled={!nameInput.trim() || nameInput.trim() === displayName}
            >
              Save
            </Button>
          </div>
        </div>
      </Card>

      {/* Export */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <Download className="w-5 h-5 text-ink-soft" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-ink">Export your work</h2>
            <p className="text-xs text-ink-faint mt-0.5">Take your notes, flashcards and study plan anywhere.</p>
          </div>
        </div>
        <Select
          value={exportCourse}
          options={courseOptions}
          onChange={setExportCourse}
          placeholder={courseOptions.length ? 'Choose a course' : 'No courses yet'}
          disabled={!courseOptions.length}
          ariaLabel="Course to export from"
        />
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {EXPORTS.map((e) => (
            <Button
              key={e.kind}
              variant="secondary"
              size="sm"
              className="justify-center"
              leftIcon={<Download className="w-3.5 h-3.5" />}
              loading={exporting === e.kind}
              disabled={!exportCourse || exporting !== null}
              onClick={() => handleExport(e.kind)}
            >
              {e.label}
            </Button>
          ))}
        </div>
      </Card>

      {/* Integrations pointer */}
      <Card padding="md" className="space-y-3">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <Link2 className="w-5 h-5 text-ink-soft" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-ink">Canvas LMS & integrations</h2>
            <p className="text-xs text-ink-faint mt-0.5">
              Imports are per-course: open a course → Materials → Integrations to pull in your syllabus, files and exam dates.
            </p>
          </div>
        </div>
      </Card>

      {/* Sign out */}
      <Card padding="md" className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-ink">Sign out</h2>
          <p className="text-xs text-ink-faint mt-0.5">You can sign back in any time — your courses stay put.</p>
        </div>
        <Button variant="secondary" leftIcon={<LogOut className="w-4 h-4" />} onClick={handleSignOut}>
          Sign out
        </Button>
      </Card>
    </div>
  )
}
