import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Users, Search, Loader2, ArrowRight } from 'lucide-react'
import { joinCourse, browseSharedCourses, type SharedCourse } from '@/lib/api'
import { useUser } from '@/hooks/useUser'
import { showError, showSuccess } from '@/lib/toast'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'

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

  const inputClass =
    'flex-1 px-3 py-2 bg-surface border border-line rounded-lg text-ink placeholder-ink-faint ' +
    'focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none text-sm transition-colors'

  return (
    <Card padding="md">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-xl bg-accent-wash border border-accent-line flex items-center justify-center">
          <Users className="w-5 h-5 text-accent" />
        </div>
        <h2 className="text-sm font-semibold text-ink">Join a class</h2>
      </div>
      <div className="flex gap-2">
        <input
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void join(code) }}
          placeholder="Enter a class share code"
          className={`${inputClass} font-mono tracking-widest uppercase`}
        />
        <Button
          onClick={() => void join(code)}
          disabled={joining || !code.trim()}
          leftIcon={joining ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
        >
          Join
        </Button>
      </div>

      <button
        onClick={() => { setBrowseOpen(!browseOpen); if (!browseOpen && results.length === 0) void search() }}
        className="text-xs text-accent hover:text-accent-deep mt-2.5 flex items-center gap-1 transition-colors"
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
              className={inputClass}
            />
            <Button variant="secondary" onClick={() => void search()}>
              {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
            </Button>
          </div>
          {results.length === 0 ? (
            <p className="text-xs text-ink-soft py-2">
              No published classes match. If a classmate shared a code with you, enter it above — or publish your
              own course from its Materials page so others can join.
            </p>
          ) : (
            results.map((c) => (
              <Card
                key={c.course_id}
                interactive
                padding="sm"
                onClick={() => void join(c.share_code)}
                className="group"
              >
                <div className="flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-ink truncate">{c.title}</div>
                    <div className="text-xs text-ink-faint truncate">
                      {[c.subject, c.school, c.term].filter(Boolean).join(' · ') || 'Shared class'}
                    </div>
                  </div>
                  <span className="text-xs text-ink-faint flex-shrink-0 ml-2">{c.join_count} joined</span>
                </div>
              </Card>
            ))
          )}
        </div>
      )}
    </Card>
  )
}
