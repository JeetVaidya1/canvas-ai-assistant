import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'
import { Calendar, Target } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'

/** Disciplined chart palette — series stay cyan/blue; grid + axis stay quiet. */
const SERIES_CYAN = '#22d3ee'
const SERIES_BLUE = '#3b82f6'
const GRID_COLOR = '#1f2738'
const TICK_STYLE = { fill: '#71717a', fontSize: 11 } // zinc-500, 11px

const TOOLTIP_STYLE = {
  background: '#1f2738',
  border: '1px solid #252e42',
  borderRadius: 8,
  fontSize: 12,
  color: '#f4f4f5', // zinc-100
}

const CHART_HEIGHT = 220

function shortDate(iso: string): string {
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

interface TrendChartsProps {
  analytics: LearningAnalytics
}

/**
 * Study-time + confidence trends. One subtle area fill (study time) is the
 * only gradient allowed in charts; confidence stays a plain blue line.
 */
export function TrendCharts({ analytics }: TrendChartsProps) {
  const trend = analytics.study_time_trend

  if (trend.length === 0) {
    return (
      <Card padding="lg">
        <SectionHead icon={Calendar} title="Study activity" />
        <EmptyState
          icon={<Calendar />}
          title="Study to see your trend"
          description="Your daily study time and confidence will chart here as you work."
        />
      </Card>
    )
  }

  const timeData = trend.map((d) => ({
    date: shortDate(d.date),
    minutes: d.duration_minutes ?? 0,
  }))
  const confData = trend.map((d) => ({
    date: shortDate(d.date),
    confidence: Math.round((d.avg_confidence ?? 0) * 100),
  }))

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card padding="lg">
        <SectionHead icon={Calendar} title="Study time" />
        <div style={{ height: CHART_HEIGHT }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeData} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
              <defs>
                <linearGradient id="studyTimeFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={SERIES_CYAN} stopOpacity={0.22} />
                  <stop offset="100%" stopColor={SERIES_CYAN} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
              <XAxis dataKey="date" tick={TICK_STYLE} stroke={GRID_COLOR} />
              <YAxis tick={TICK_STYLE} stroke={GRID_COLOR} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value) => [`${value as number} min`, 'Study time']}
              />
              <Area
                type="monotone"
                dataKey="minutes"
                stroke={SERIES_CYAN}
                strokeWidth={2}
                fill="url(#studyTimeFill)"
                dot={{ r: 3, fill: SERIES_CYAN }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card padding="lg">
        <SectionHead icon={Target} title="Confidence over time" chip="bg-blue-500/12 border-blue-400/20" tint="text-sky-300" />
        <div style={{ height: CHART_HEIGHT }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={confData} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
              <XAxis dataKey="date" tick={TICK_STYLE} stroke={GRID_COLOR} />
              <YAxis domain={[0, 100]} unit="%" tick={TICK_STYLE} stroke={GRID_COLOR} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value) => [`${value as number}%`, 'Avg confidence']}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke={SERIES_BLUE}
                strokeWidth={2}
                dot={{ r: 3, fill: SERIES_BLUE }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
