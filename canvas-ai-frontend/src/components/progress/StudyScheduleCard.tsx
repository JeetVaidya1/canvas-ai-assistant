import { Calendar } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { SectionHead } from './SectionHead'

const QUESTIONS_FOR_EXAM_HINT = 50

interface StudyScheduleCardProps {
  analytics: LearningAnalytics
}

/** Suggested study cadence derived from the current analytics snapshot. */
export function StudyScheduleCard({ analytics }: StudyScheduleCardProps) {
  const items = [
    {
      heading: 'Today',
      body:
        analytics.weak_areas.length > 0
          ? `Review ${analytics.weak_areas[0]} concepts`
          : 'Great job! Try exploring new topics',
    },
    {
      heading: 'This week',
      body: 'Practice problems on your strongest topics to maintain mastery',
    },
    {
      heading: 'Next steps',
      body:
        analytics.total_questions < QUESTIONS_FOR_EXAM_HINT
          ? 'Ask more questions to get better insights'
          : 'Consider taking a practice exam',
    },
  ]

  return (
    <Card accent padding="lg">
      <SectionHead icon={Calendar} title="Suggested study schedule" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {items.map((item) => (
          <div key={item.heading} className="bg-white/[0.04] rounded-lg p-4 border border-white/10">
            <h3 className="text-xs font-semibold text-gradient-brand mb-2">{item.heading}</h3>
            <p className="text-sm text-zinc-300">{item.body}</p>
          </div>
        ))}
      </div>
    </Card>
  )
}
