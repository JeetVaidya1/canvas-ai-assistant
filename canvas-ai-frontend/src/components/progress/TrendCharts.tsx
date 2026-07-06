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
import { Calendar } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'

/** Paper & Ink chart palette — accent (primary series) + ink (secondary);
 *  grid/axis and ticks stay recessive on the white card. */
const SERIES_ACCENT = '#2b4acb'
const SERIES_INK = '#211f1a'
const GRID_COLOR = '#e7e3d9'
const TICK_STYLE = { fill: '#8d877b', fontSize: 11 } // ink-faint, 11px

// Tooltip = white card: paper sheet, hairline border, ink text.
const TOOLTIP_STYLE = {
  background: '#ffffff',
  border: '1px solid #e7e3d9',
  borderRadius: 8,
  fontSize: 12,
  color: '#211f1a',
  boxShadow: '0 2px 8px rgba(33, 31, 26, 0.08)',
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
 * Study-time + confidence trends. Study time carries the page's single
 * low-opacity accent area fill; confidence stays a plain ink line.
 */
export function TrendCharts({ analytics }: TrendChartsProps) {
  const trend = analytics.study_time_trend

  if (trend.length === 0) {
    return (
      <Card padding="lg">
        <SectionHead num="06" title="Study activity" />
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
        <SectionHead num="06" title="Study time" />
        <div style={{ height: CHART_HEIGHT }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeData} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
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
                stroke={SERIES_ACCENT}
                strokeWidth={2}
                fill={SERIES_ACCENT}
                fillOpacity={0.1}
                dot={{ r: 3, fill: SERIES_ACCENT }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card padding="lg">
        <SectionHead num="07" title="Confidence over time" />
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
                stroke={SERIES_INK}
                strokeWidth={2}
                dot={{ r: 3, fill: SERIES_INK }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
