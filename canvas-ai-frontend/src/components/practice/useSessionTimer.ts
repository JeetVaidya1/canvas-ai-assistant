import { useCallback, useEffect, useState } from 'react'

/**
 * Second-resolution session timer. Ticks while `active` is true, freezes when
 * it flips false (e.g. when results are shown), and resets on demand.
 */
export function useSessionTimer(active: boolean): { timeElapsed: number; reset: () => void } {
  const [timeElapsed, setTimeElapsed] = useState(0)

  useEffect(() => {
    if (!active) return
    const interval = window.setInterval(() => setTimeElapsed((prev) => prev + 1), 1000)
    return () => window.clearInterval(interval)
  }, [active])

  const reset = useCallback(() => setTimeElapsed(0), [])

  return { timeElapsed, reset }
}
