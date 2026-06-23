import { useState, useCallback, useEffect } from 'react'

interface RecentEntry {
  courseId: string
  page: string
  timestamp: number
}

const STORAGE_KEY = 'vindexa-recent-activity'
const MAX_ENTRIES = 5

function readEntries(): RecentEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw) as RecentEntry[]
  } catch {
    return []
  }
}

function writeEntries(entries: RecentEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
}

export function trackVisit(courseId: string, page: string) {
  const entries = readEntries()
  // Remove existing entry for same courseId+page
  const filtered = entries.filter(
    (e) => !(e.courseId === courseId && e.page === page)
  )
  // Prepend new entry
  const updated = [{ courseId, page, timestamp: Date.now() }, ...filtered].slice(
    0,
    MAX_ENTRIES
  )
  writeEntries(updated)
}

export function useRecentActivity() {
  const [entries, setEntries] = useState<RecentEntry[]>(readEntries)

  const refresh = useCallback(() => {
    setEntries(readEntries())
  }, [])

  // Refresh on mount and when storage changes from another tab
  useEffect(() => {
    refresh()
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) refresh()
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [refresh])

  return entries
}
