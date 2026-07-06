/** Pulse skeleton mirroring the Analytics layout: hero, stat strip, tiles. */
export function AnalyticsSkeleton() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6 animate-pulse" aria-hidden>
      {/* Hero */}
      <div className="h-40 rounded-xl bg-paper-deep border border-line" />
      {/* Stat strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-28 rounded-xl bg-paper-deep border border-line" />
        ))}
      </div>
      {/* Concept map */}
      <div className="h-64 rounded-xl bg-paper-deep border border-line" />
      {/* Bento */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-72 rounded-xl bg-paper-deep border border-line" />
        <div className="h-72 rounded-xl bg-paper-deep border border-line" />
      </div>
    </div>
  )
}
