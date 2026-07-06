interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
  label?: string
}

export default function LoadingSpinner({ size = 'md', className = '', label }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-5 h-5 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-3',
  }

  return (
    <div className={`flex flex-col items-center justify-center gap-3 ${className}`}>
      <div
        className={`${sizeClasses[size]} border-line-strong border-t-accent rounded-full animate-spin`}
      />
      {label && <p className="text-sm text-ink-soft">{label}</p>}
    </div>
  )
}
