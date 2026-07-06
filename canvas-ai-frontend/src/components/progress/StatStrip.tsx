import { Flame, BookOpen, Target, Brain, type LucideIcon } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import CountUp from '@/components/ui/CountUp'

interface StatTileProps {
  icon: LucideIcon
  tint: string
  chip: string
  label: string
  value: number
  suffix?: string
  unit: string
}

function StatTile({ icon: Icon, tint, chip, label, value, suffix = '', unit }: StatTileProps) {
  return (
    <Card accent elevation={1}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-zinc-400">{label}</p>
          <p className="text-3xl font-bold text-zinc-50 mt-1.5 leading-none tabular-nums">
            <CountUp to={value} />{suffix}
          </p>
          <p className="text-xs text-zinc-500 mt-1.5">{unit}</p>
        </div>
        <div className={`w-10 h-10 rounded-xl border flex items-center justify-center ${chip}`}>
          <Icon className={`w-5 h-5 ${tint}`} />
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
        icon={Flame}
        tint="text-cyan-300"
        chip="bg-cyan-500/12 border-cyan-400/20"
        label="Study streak"
        value={analytics.study_streak}
        unit="days"
      />
      <StatTile
        icon={BookOpen}
        tint="text-sky-300"
        chip="bg-blue-500/12 border-blue-400/20"
        label="Questions asked"
        value={analytics.total_questions}
        unit="total"
      />
      <StatTile
        icon={Target}
        tint="text-emerald-300"
        chip="bg-emerald-500/12 border-emerald-400/20"
        label="Avg confidence"
        value={Math.round(analytics.avg_confidence * 100)}
        suffix="%"
        unit="score"
      />
      <StatTile
        icon={Brain}
        tint="text-sky-300"
        chip="bg-sky-500/12 border-sky-400/20"
        label="Topics studied"
        value={analytics.topics_progress.length}
        unit="concepts"
      />
    </div>
  )
}
