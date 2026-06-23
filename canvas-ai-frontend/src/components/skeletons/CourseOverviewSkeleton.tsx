import Skeleton from '@/components/shared/Skeleton'

export default function CourseOverviewSkeleton() {
  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      {/* Title */}
      <div>
        <Skeleton className="h-7 w-64 mb-2" />
        <Skeleton className="h-4 w-32" />
      </div>

      {/* Quick actions */}
      <div className="flex items-center gap-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-24 rounded-lg" />
        ))}
      </div>

      {/* Files section */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 space-y-3">
        <Skeleton className="h-4 w-20 mb-4" />
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-10 w-full" />
        ))}
      </div>
    </div>
  )
}
