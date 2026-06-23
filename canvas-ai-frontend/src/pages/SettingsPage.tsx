import { useState } from 'react'
import { Download, Upload, Activity, User } from 'lucide-react'
import { useProfile } from '@/hooks/useProfile'
import { showSuccess } from '@/lib/toast'

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
      <h1 className="text-xl font-semibold text-zinc-100">Settings</h1>

      {/* Profile */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-medium text-zinc-100 flex items-center gap-2">
          <User className="w-4 h-4 text-zinc-400" />
          Profile
        </h2>
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-cyan-600 flex items-center justify-center text-lg font-semibold text-white flex-shrink-0">
            {initial}
          </div>
          <div className="flex-1 space-y-2">
            <input
              type="text"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              placeholder="Display name"
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none text-sm"
            />
            <button
              onClick={handleSaveProfile}
              disabled={!nameInput.trim() || nameInput.trim() === displayName}
              className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
          </div>
        </div>
      </div>

      {/* LMS Import */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-medium text-zinc-100 flex items-center gap-2">
          <Upload className="w-4 h-4 text-zinc-400" />
          Canvas LMS Import
        </h2>
        <p className="text-xs text-zinc-500">
          Import courses and materials directly from Canvas LMS.
        </p>
        <div className="space-y-3">
          <input
            type="text"
            placeholder="Canvas API Token"
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none text-sm"
          />
          <input
            type="text"
            placeholder="Canvas Course ID"
            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 placeholder-zinc-500 focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 outline-none text-sm"
          />
          <button className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
            Import from Canvas
          </button>
        </div>
      </div>

      {/* Export */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-medium text-zinc-100 flex items-center gap-2">
          <Download className="w-4 h-4 text-zinc-400" />
          Export Tools
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <button className="flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-zinc-300 text-xs font-medium transition-colors">
            <Download className="w-3.5 h-3.5" />
            Export Notes (PDF)
          </button>
          <button className="flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-zinc-300 text-xs font-medium transition-colors">
            <Download className="w-3.5 h-3.5" />
            Export Flashcards (Anki)
          </button>
          <button className="flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-zinc-300 text-xs font-medium transition-colors">
            <Download className="w-3.5 h-3.5" />
            Export Planner (iCal)
          </button>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-medium text-zinc-100 flex items-center gap-2">
          <Activity className="w-4 h-4 text-zinc-400" />
          System Status
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="flex items-center gap-3 px-3 py-2 bg-zinc-800/50 rounded-lg">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-zinc-300">API Server</span>
            <span className="text-xs text-zinc-500 ml-auto">Connected</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2 bg-zinc-800/50 rounded-lg">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-zinc-300">Database</span>
            <span className="text-xs text-zinc-500 ml-auto">Connected</span>
          </div>
        </div>
      </div>
    </div>
  )
}
