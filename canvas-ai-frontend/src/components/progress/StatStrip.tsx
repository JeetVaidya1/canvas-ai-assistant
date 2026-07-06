import { CalendarDays, BookOpen, Target, BarChart3, type LucideIcon } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import CountUp from '@/components/ui/CountUp'

interface StatTileProps {
  icon: LucideIcon
  label: string
  value: number
  suffix?: string
  unit: string
}

/** Report-card stat tile: serif display number, quiet ink icon chip. */
function StatTile({ icon: Icon, label, value, suffix = '', unit }: StatTileProps) {
  return (
    <Card elevation={1}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-ink-soft">{label}</p>
          <p className="font-display text-3xl font-semibold text-ink mt-1.5 leading-none tnum">
            <CountUp to={value} />{suffix}
          </p>
          <p className="text-xs text-ink-faint mt-1.5">{unit}</p>
        </div>
        <div className="w-10 h-10 rounded-lg bg-paper-deep border border-line flex items-center justify-center">
          <Icon className="w-5 h-5 text-ink-soft" />
        </div>
      </div>
    </Card>
  )
}

interface StatStripProps {
  analytics: LearningAnalytics
}

/** Four CountUp stat tiles: streak, questions, confidence, topics. */
export function StatStrip({ analytics }: StatStripProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatTile
        icon={CalendarDays}
        label="Study streak"
        value={analytics.study_streak}
        unit="days"
      />
      <StatTile
        icon={BookOpen}
        label="Questions asked"
        value={analytics.total_questions}
        unit="total"
      />
      <StatTile
        icon={Target}
        label="Avg confidence"
        value={Math.round(analytics.avg_confidence * 100)}
        suffix="%"
        unit="score"
      />
      <StatTile
        icon={BarChart3}
        label="Topics studied"
        value={analytics.topics_progress.length}
        unit="concepts"
      />
    </div>
  )
}
