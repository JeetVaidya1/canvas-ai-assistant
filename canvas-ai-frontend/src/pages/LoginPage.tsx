import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { Mail, Lock } from 'lucide-react'
import { supabase } from '@/lib/supabaseClient'
import { useAuth } from '@/lib/auth'
import { showError, showSuccess } from '@/lib/toast'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'

type Mode = 'signin' | 'signup' | 'magic'

const TITLES: Record<Mode, string> = {
  signin: 'Welcome back',
  signup: 'Create your account',
  magic: 'Sign in with a magic link',
}

const SUBMIT_LABELS: Record<Mode, string> = {
  signin: 'Sign in',
  signup: 'Create account',
  magic: 'Send magic link',
}

export default function LoginPage() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [sent, setSent] = useState(false)

  if (loading) return null
  if (user) return <Navigate to="/dashboard" replace />

  const redirectTo = `${window.location.origin}/dashboard`

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    try {
      if (mode === 'magic') {
        const { error } = await supabase.auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo } })
        if (error) throw error
        setSent(true)
      } else if (mode === 'signup') {
        const { error } = await supabase.auth.signUp({ email, password, options: { emailRedirectTo: redirectTo } })
        if (error) throw error
        setSent(true)
        showSuccess('Account created — check your email to confirm, then sign in.')
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        // onAuthStateChange flips the route to /dashboard.
      }
    } catch (err) {
      showError(err instanceof Error ? err.message : 'Authentication failed')
    } finally {
      setBusy(false)
    }
  }

  const google = async () => {
    const { error } = await supabase.auth.signInWithOAuth({ provider: 'google', options: { redirectTo } })
    if (error) showError(error.message)
  }

  return (
    <div className="min-h-screen bg-paper flex items-center justify-center px-4">
      <div className="w-full max-w-sm animate-fade-up">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2.5 mb-8 mx-auto focus-ring rounded-lg"
          aria-label="Back to home"
        >
          <BrandMark className="w-8 h-8" />
          <span className="font-display text-lg font-semibold text-ink tracking-tight">Vindexa</span>
        </button>

        <div className="card-surface elev-2 rounded-2xl p-7">
          {sent ? (
            <div className="text-center py-6">
              <div className="w-12 h-12 rounded-xl bg-accent-wash border border-accent-line flex items-center justify-center mx-auto mb-4">
                <Mail className="w-5 h-5 text-accent" />
              </div>
              <h2 className="font-display text-lg font-semibold text-ink mb-1">Check your email</h2>
              <p className="text-sm text-ink-soft">
                We sent a link to <span className="text-ink font-medium">{email}</span>.
              </p>
              <Button variant="ghost" size="sm" className="mt-5" onClick={() => { setSent(false); setMode('signin') }}>
                Back to sign in
              </Button>
            </div>
          ) : (
            <>
              <h1 className="font-display text-2xl font-semibold text-ink mb-1">{TITLES[mode]}</h1>
              <p className="text-sm text-ink-soft mb-6">Your courses and progress, private to you.</p>

              <button
                onClick={() => void google()}
                className="w-full mb-4 bg-surface text-ink border border-line py-2.5 rounded-lg font-medium text-sm hover:bg-surface-hover hover:border-line-strong transition-colors flex items-center justify-center gap-2 focus-ring"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" aria-hidden="true"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.99.66-2.26 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0012 23z"/><path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 010-4.2V7.06H2.18a11 11 0 000 9.88l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84C6.71 7.3 9.14 5.38 12 5.38z"/></svg>
                Continue with Google
              </button>

              <div className="flex items-center gap-3 mb-4">
                <div className="flex-1 h-px bg-line" />
                <span className="text-xs text-ink-faint">or</span>
                <div className="flex-1 h-px bg-line" />
              </div>

              <form onSubmit={submit} className="space-y-3">
                <Input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@university.edu"
                  leftIcon={<Mail />}
                  aria-label="Email"
                  autoComplete="email"
                />
                {mode !== 'magic' && (
                  <Input
                    type="password"
                    required
                    minLength={6}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    leftIcon={<Lock />}
                    aria-label="Password"
                    autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                  />
                )}
                <Button type="submit" loading={busy} className="w-full justify-center">
                  {SUBMIT_LABELS[mode]}
                </Button>
              </form>

              <div className="mt-5 text-center text-sm text-ink-faint space-y-1.5">
                {mode !== 'magic' && (
                  <button onClick={() => setMode('magic')} className="text-accent hover:text-accent-deep block w-full transition-colors">
                    Email me a magic link instead
                  </button>
                )}
                {mode === 'signin' && (
                  <button onClick={() => setMode('signup')} className="hover:text-ink-soft block w-full transition-colors">
                    New here? <span className="text-accent hover:text-accent-deep">Create an account</span>
                  </button>
                )}
                {mode !== 'signin' && (
                  <button onClick={() => setMode('signin')} className="hover:text-ink-soft block w-full transition-colors">
                    Already have an account? <span className="text-accent hover:text-accent-deep">Sign in</span>
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
