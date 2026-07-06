import { useNavigate } from 'react-router-dom'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'

export default function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <div className="flex flex-col items-center justify-center h-full py-20 px-6 text-center">
      <BrandMark className="mb-6 h-14 w-14" />
      <div className="font-display text-7xl font-semibold text-ink mb-4">404</div>
      <h1 className="text-xl font-semibold text-ink mb-2">Page not found</h1>
      <p className="text-sm text-ink-soft mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Button size="lg" onClick={() => navigate('/dashboard')}>
        Back to Dashboard
      </Button>
    </div>
  )
}
