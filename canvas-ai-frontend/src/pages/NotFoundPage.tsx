import { useNavigate } from 'react-router-dom'

export default function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <div className="flex flex-col items-center justify-center h-full py-20 px-6 text-center">
      <div className="text-7xl font-bold text-zinc-700 mb-4">404</div>
      <h1 className="text-xl font-semibold text-zinc-100 mb-2">Page not found</h1>
      <p className="text-sm text-zinc-500 mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <button
        onClick={() => navigate('/dashboard')}
        className="bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
      >
        Back to Dashboard
      </button>
    </div>
  )
}
