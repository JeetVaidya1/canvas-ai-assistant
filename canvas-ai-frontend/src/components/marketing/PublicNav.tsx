import { Link, useNavigate } from 'react-router-dom'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'

/**
 * Shared fixed top nav for all public (unauthenticated) pages.
 * Solid paper with a hairline below — pages should pad the top
 * (e.g. `pt-32`) to clear the 14-unit bar.
 */
export function PublicNav() {
  const navigate = useNavigate()
  const toLogin = () => navigate('/login')

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 top-bar">
      <div className="flex items-center justify-between px-6 h-14 max-w-6xl mx-auto">
        <Link to="/" className="flex items-center gap-2.5 rounded-md focus-ring">
          <BrandMark className="w-7 h-7" />
          <span className="font-display text-lg font-semibold text-ink tracking-tight">Vindexa</span>
        </Link>
        <div className="flex items-center gap-1 sm:gap-2">
          <Link
            to="/pricing"
            className="hidden sm:inline-block text-sm text-ink-soft hover:text-ink px-3 py-1.5 rounded-lg transition-colors focus-ring"
          >
            Pricing
          </Link>
          <Link
            to="/help"
            className="hidden sm:inline-block text-sm text-ink-soft hover:text-ink px-3 py-1.5 rounded-lg transition-colors focus-ring"
          >
            Help
          </Link>
          <Button variant="ghost" size="sm" onClick={toLogin}>Sign in</Button>
          <Button size="sm" onClick={toLogin}>Get started</Button>
        </div>
      </div>
    </nav>
  )
}
