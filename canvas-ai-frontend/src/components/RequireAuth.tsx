import { useEffect, useRef } from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '@/lib/auth'
import { claimLegacyData } from '@/lib/api'
import LoadingSpinner from '@/components/shared/LoadingSpinner'

export default function RequireAuth() {
  const { user, loading } = useAuth()
  const claimed = useRef(false)

  // First sign-in claims any unowned legacy courses for this account (idempotent).
  useEffect(() => {
    if (user && !claimed.current) {
      claimed.current = true
      void claimLegacyData().catch(() => {})
    }
  }, [user])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner size="lg" />
      </div>
    )
  }
  if (!user) return <Navigate to="/login" replace />
  return <Outlet />
}
