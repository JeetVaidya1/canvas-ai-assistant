import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { Loader2, Mail, Lock, GraduationCap } from 'lucide-react'
import { supabase } from '@/lib/supabaseClient'
import { useAuth } from '@/lib/auth'
import { showError, showSuccess } from '@/lib/toast'

type Mode = 'signin' | 'signup' | 'magic'

export default function LoginPage() {
  const { user, loading } = useAuth()
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
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-zinc-900/80 backdrop-blur border border-zinc-800 rounded-2xl p-7">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/15 flex items-center justify-center">
            <GraduationCap className="w-5 h-5 text-cyan-400" />
          </div>
          <span className="text-lg font-semibold text-zinc-50">Vindexa</span>
        </div>

        {sent ? (
          <div className="text-center py-6">
            <Mail className="w-10 h-10 text-cyan-400 mx-auto mb-3" />
            <h2 className="text-lg font-semibold text-zinc-100 mb-1">Check your email</h2>
            <p className="text-sm text-zinc-500">We sent a link to <span className="text-zinc-300">{email}</span>.</p>
            <button onClick={() => { setSent(false); setMode('signin') }} className="text-cyan-400 text-sm mt-4 hover:text-cyan-300">
              Back to sign in
            </button>
          </div>
        ) : (
          <>
            <h1 className="text-xl font-semibold text-zinc-50 mb-1">
              {mode === 'signup' ? 'Create your account' : mode === 'magic' ? 'Magic link sign in' : 'Welcome back'}
            </h1>
            <p className="text-sm text-zinc-500 mb-5">Your courses and progress, private to you.</p>

            <button
              onClick={() => void google()}
              className="w-full mb-4 bg-white text-zinc-800 py-2.5 rounded-lg font-medium text-sm hover:bg-zinc-100 transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.99.66-2.26 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0012 23z"/><path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 010-4.2V7.06H2.18a11 11 0 000 9.88l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84C6.71 7.3 9.14 5.38 12 5.38z"/></svg>
              Continue with Google
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="flex-1 h-px bg-zinc-800" /><span className="text-xs text-zinc-600">or</span><div className="flex-1 h-px bg-zinc-800" />
            </div>

            <form onSubmit={submit} className="space-y-3">
              <div className="relative">
                <Mail className="w-4 h-4 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email"
                  className="w-full pl-9 pr-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 text-sm focus:border-cyan-600 outline-none"
                />
              </div>
              {mode !== 'magic' && (
                <div className="relative">
                  <Lock className="w-4 h-4 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password"
                    className="w-full pl-9 pr-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-100 text-sm focus:border-cyan-600 outline-none"
                  />
                </div>
              )}
              <button
                type="submit" disabled={busy}
                className="w-full bg-cyan-600 text-white py-2.5 rounded-lg font-medium text-sm hover:bg-cyan-500 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {busy && <Loader2 className="w-4 h-4 animate-spin" />}
                {mode === 'signup' ? 'Create account' : mode === 'magic' ? 'Send magic link' : 'Sign in'}
              </button>
            </form>

            <div className="mt-5 text-center text-sm text-zinc-500 space-y-1.5">
              {mode !== 'magic' && (
                <button onClick={() => setMode('magic')} className="text-cyan-400 hover:text-cyan-300 block w-full">Email me a magic link instead</button>
              )}
              {mode === 'signin' && (
                <button onClick={() => setMode('signup')} className="hover:text-zinc-300 block w-full">New here? <span className="text-cyan-400">Create an account</span></button>
              )}
              {mode !== 'signin' && (
                <button onClick={() => setMode('signin')} className="hover:text-zinc-300 block w-full">Already have an account? <span className="text-cyan-400">Sign in</span></button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
