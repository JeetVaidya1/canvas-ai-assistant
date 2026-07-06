import { TrendingUp, Brain } from 'lucide-react'
import type { Readiness } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { ProgressBar } from '@/components/ui/Progress'
import { EmptyState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'

const MAX_TOPICS = 12

interface TopicMasteryListProps {
  readiness: Readiness | null
}

/**
 * Per-topic mastery breakdown from readiness.by_topic — each bar is colored by
 * the semantic scoreTone so weak topics read rose, solid topics emerald.
 */
export function TopicMasteryList({ readiness }: TopicMasteryListProps) {
  const topics = (readiness?.by_topic ?? []).filter((t) => t.has_data).slice(0, MAX_TOPICS)

  return (
    <Card padding="lg" className="h-full">
      <SectionHead
        icon={TrendingUp}
        title="Topic mastery"
        hint="Where your readiness score comes from"
      />
      {topics.length === 0 ? (
        <EmptyState
          icon={<Brain />}
          title="No topic mastery yet"
          description="Ask questions or take a quiz to start scoring topics."
        />
      ) : (
        <div className="space-y-3.5">
          {topics.map((t) => {
            const pct = Math.round(t.mastery_pct)
            return (
              <div key={t.topic}>
                <div className="flex items-center justify-between text-sm mb-1.5">
                  <span className="text-zinc-200 truncate pr-3">{t.topic}</span>
                  <span className="text-zinc-100 tabular-nums font-semibold">{pct}%</span>
                </div>
                <ProgressBar value={t.mastery_pct} className="h-2.5" label={`${t.topic} mastery`} />
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}
