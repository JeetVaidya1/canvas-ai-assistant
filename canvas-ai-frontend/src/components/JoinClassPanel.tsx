import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Users, Search, Loader2, ArrowRight } from 'lucide-react'
import { joinCourse, browseSharedCourses, type SharedCourse } from '@/lib/api'
import { useUser } from '@/hooks/useUser'
import { showError, showSuccess } from '@/lib/toast'

export default function JoinClassPanel() {
  const userId = useUser()
  const navigate = useNavigate()
  const [code, setCode] = useState('')
  const [joining, setJoining] = useState(false)
  const [browseOpen, setBrowseOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SharedCourse[]>([])
  const [searching, setSearching] = useState(false)

  const join = async (shareCode: string) => {
    const c = shareCode.trim().toUpperCase()
    if (!c) return
    setJoining(true)
    try {
      const r = await joinCourse(c, userId)
      showSuccess(r.newly_joined ? `Joined ${r.title}` : `You're already in ${r.title}`)
      navigate(`/course/${r.course_id}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Could not join — check the code')
    } finally {
      setJoining(false)
    }
  }

  const search = async () => {
    setSearching(true)
    try {
      setResults(await browseSharedCourses(query))
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Users className="w-4 h-4 text-cyan-400" />
        <h2 className="text-sm font-semibold text-zinc-100">Join a class</h2>
      </div>
      <div className="flex gap-2">
        <input
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void join(code) }}
          placeholder="Enter a class share code"
          className="flex-1 px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm font-mono tracking-widest uppercase"
        />
        <button
          onClick={() => void join(code)}
          disabled={joining || !code.trim()}
          className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2"
        >
          {joining ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
          Join
        </button>
      </div>

      <button
        onClick={() => { setBrowseOpen(!browseOpen); if (!browseOpen && results.length === 0) void search() }}
        className="text-xs text-cyan-400 hover:text-cyan-300 mt-2.5 flex items-center gap-1"
      >
        <Search className="w-3 h-3" /> {browseOpen ? 'Hide catalog' : 'Browse shared classes'}
      </button>

      {browseOpen && (
        <div className="mt-3 space-y-2">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void search() }}
              placeholder="Search by subject, school, or title"
              className="flex-1 px-3 py-1.5 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
            <button onClick={() => void search()} className="px-3 py-1.5 border border-zinc-700 rounded-lg text-sm text-zinc-300 hover:bg-zinc-800">
              {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
            </button>
          </div>
          {results.length === 0 ? (
            <p className="text-xs text-zinc-500 py-2">No published classes match yet.</p>
          ) : (
            results.map((c) => (
              <button
                key={c.course_id}
                onClick={() => void join(c.share_code)}
                className="w-full text-left p-3 rounded-lg border border-zinc-700/50 bg-zinc-900/50 hover:bg-zinc-800 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-zinc-200 truncate">{c.title}</div>
                    <div className="text-xs text-zinc-500 truncate">
                      {[c.subject, c.school, c.term].filter(Boolean).join(' · ') || 'Shared class'}
                    </div>
                  </div>
                  <span className="text-xs text-zinc-500 flex-shrink-0 ml-2">{c.join_count} joined</span>
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
