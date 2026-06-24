import { useState } from 'react'
import { Download, Upload, Activity, User } from 'lucide-react'
import { useProfile } from '@/hooks/useProfile'
import { showSuccess } from '@/lib/toast'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'

const inputClass =
  'w-full px-3 py-2 bg-zinc-800/70 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-600 ' +
  'focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-sm transition-colors'

const sectionIconClass =
  'w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0'

export default function SettingsPage() {
  const { displayName, setDisplayName } = useProfile()
  const [nameInput, setNameInput] = useState(displayName)

  const handleSaveProfile = () => {
    setDisplayName(nameInput.trim())
    showSuccess('Display name updated')
  }

  const initial = (displayName || 'U').charAt(0).toUpperCase()

  return (
    <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
      <PageHeader eyebrow="Settings" title="Settings" />

      {/* Profile */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <User className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-100">Profile</h2>
        </div>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-gradient-brand flex items-center justify-center text-lg font-semibold text-white flex-shrink-0 glow-brand-sm">
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

      {/* LMS Import */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <Upload className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-100">Canvas LMS Import</h2>
        </div>
        <p className="text-xs text-zinc-500">
          Import courses and materials directly from Canvas LMS.
        </p>
        <div className="space-y-3">
          <input
            type="text"
            placeholder="Canvas API Token"
            className={inputClass}
          />
          <input
            type="text"
            placeholder="Canvas Course ID"
            className={inputClass}
          />
          <Button leftIcon={<Upload className="w-4 h-4" />}>
            Import from Canvas
          </Button>
        </div>
      </Card>

      {/* Export */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <Download className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-100">Export Tools</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Button variant="secondary" size="sm" className="justify-center" leftIcon={<Download className="w-3.5 h-3.5" />}>
            Export Notes (PDF)
          </Button>
          <Button variant="secondary" size="sm" className="justify-center" leftIcon={<Download className="w-3.5 h-3.5" />}>
            Export Flashcards (Anki)
          </Button>
          <Button variant="secondary" size="sm" className="justify-center" leftIcon={<Download className="w-3.5 h-3.5" />}>
            Export Planner (iCal)
          </Button>
        </div>
      </Card>

      {/* System Status */}
      <Card padding="md" className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={sectionIconClass}>
            <Activity className="w-5 h-5 text-cyan-300" />
          </div>
          <h2 className="text-sm font-semibold text-zinc-100">System Status</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="flex items-center gap-3 px-3 py-2 bg-zinc-800/50 rounded-lg border border-zinc-700/40">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-zinc-300">API Server</span>
            <span className="text-xs text-zinc-500 ml-auto">Connected</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2 bg-zinc-800/50 rounded-lg border border-zinc-700/40">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-zinc-300">Database</span>
            <span className="text-xs text-zinc-500 ml-auto">Connected</span>
          </div>
        </div>
      </Card>
    </div>
  )
}
