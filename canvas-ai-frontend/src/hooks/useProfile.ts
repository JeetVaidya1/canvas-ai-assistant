import { useState, useCallback } from 'react'

const STORAGE_KEY = 'vindexa_display_name'

export function useProfile() {
  const [displayName, setDisplayNameState] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || ''
  })

  const setDisplayName = useCallback((name: string) => {
    localStorage.setItem(STORAGE_KEY, name)
    setDisplayNameState(name)
  }, [])

  return { displayName, setDisplayName }
}
