import { AlertCircle, Award, Zap } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'

interface FocusPanelProps {
  analytics: LearningAnalytics
}

/** Weak areas + study recommendations — stacked in a single right-rail tile. */
export function FocusPanel({ analytics }: FocusPanelProps) {
  return (
    <div className="flex flex-col gap-4 h-full">
      <Card padding="lg" className="flex-1">
        <SectionHead
          icon={AlertCircle}
          title="Areas to focus on"
          chip="bg-rose-500/10 border-rose-500/20"
          tint="text-rose-400"
        />
        {analytics.weak_areas.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-6">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-400/20 flex items-center justify-center mb-3">
              <Award className="w-6 h-6 text-emerald-300" />
            </div>
            <p className="text-emerald-300 font-medium">No weak areas</p>
            <p className="text-zinc-400 text-sm mt-0.5">Keep practicing to hold your edge.</p>
          </div>
        ) : (
          <div className="space-y-2.5">
            {analytics.weak_areas.map((area) => (
              <div key={area} className="bg-rose-500/10 border border-rose-500/25 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                  <span className="font-medium text-rose-300 capitalize">{area}</span>
                </div>
                <p className="text-rose-200/80 text-sm mt-1">Review this topic and practice more problems.</p>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card padding="lg" className="flex-1">
        <SectionHead
          icon={Zap}
          title="Recommendations"
          chip="bg-amber-500/10 border-amber-500/20"
          tint="text-amber-400"
        />
        {analytics.study_recommendations.length === 0 ? (
          <EmptyState
            icon={<Zap />}
            title="No recommendations yet"
            description="Study a bit more and we’ll suggest your next moves."
            className="py-6"
          />
        ) : (
          <div className="space-y-2.5">
            {analytics.study_recommendations.map((rec) => (
              <div key={rec} className="bg-amber-500/10 border border-amber-500/25 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                  <p className="text-amber-100/90 text-sm">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
