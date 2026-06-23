import { useAuth } from '@/lib/auth'

/**
 * The authenticated user's id. Returns '' while loading or signed out — callers
 * are rendered behind RequireAuth, so a real id is present in practice.
 * (Previously a localStorage-generated anonymous id; now the Supabase auth uid.)
 */
export function useUser(): string {
  const { user } = useAuth()
  return user?.id ?? ''
}
