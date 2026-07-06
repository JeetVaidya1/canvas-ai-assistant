import Skeleton from '@/components/shared/Skeleton'

export default function DashboardSkeleton() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
      {/* Greeting */}
      <div>
        <Skeleton className="h-8 w-56 mb-2" />
        <Skeleton className="h-4 w-40" />
      </div>

      {/* Courses header */}
      <div className="flex items-center justify-between">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-9 w-32 rounded-lg" />
      </div>

      {/* Course card grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-paper-deep border border-line rounded-xl p-5 space-y-3">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        ))}
      </div>
    </div>
  )
}
