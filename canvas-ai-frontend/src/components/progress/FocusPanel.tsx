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
        <SectionHead num="04" title="Areas to focus on" />
        {analytics.weak_areas.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-6">
            <div className="w-12 h-12 rounded-2xl bg-success-wash border border-success/25 flex items-center justify-center mb-3">
              <Award className="w-6 h-6 text-success" />
            </div>
            <p className="text-success font-medium">No weak areas</p>
            <p className="text-ink-soft text-sm mt-0.5">Keep practicing to hold your edge.</p>
          </div>
        ) : (
          <div className="space-y-2.5">
            {analytics.weak_areas.map((area) => (
              <div key={area} className="bg-danger-wash border border-danger/25 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-danger flex-shrink-0" />
                  <span className="font-medium text-danger capitalize">{area}</span>
                </div>
                <p className="text-ink-soft text-sm mt-1">Review this topic and practice more problems.</p>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card padding="lg" className="flex-1">
        <SectionHead num="05" title="Recommendations" />
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
              <div key={rec} className="bg-accent-wash border border-accent-line rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                  <p className="text-ink text-sm">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
