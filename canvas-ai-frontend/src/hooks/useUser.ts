import { useState } from 'react'

function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

const STORAGE_KEY = 'vindexa_user_id'

export function useUser() {
  const [userId] = useState<string>(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) return stored
    const id = generateId()
    localStorage.setItem(STORAGE_KEY, id)
    return id
  })

  return userId
}
