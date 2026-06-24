import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { BarChart3, Calendar } from 'lucide-react'
import AnalyticsDashboard from '@/components/AnalyticsDashboard'
import PlannerPage from '@/pages/PlannerPage'
import { SubTabs, type SubTab } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'

type ProgressTab = 'analytics' | 'planner'

const TABS: SubTab[] = [
  { key: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-4 h-4" /> },
  { key: 'planner', label: 'Planner', icon: <Calendar className="w-4 h-4" /> },
]

/**
 * Progress — consolidated destination that composes the existing
 * AnalyticsDashboard and PlannerPage behind a segmented sub-tab control.
 */
export default function ProgressPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const [tab, setTab] = useState<ProgressTab>('analytics')

  const courseTitle =
    courses?.find((c) => c.course_id === courseId)?.title ?? 'Progress'

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-5 pb-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand">
            Progress
          </p>
          <h1 className="text-lg font-semibold text-zinc-100">{courseTitle}</h1>
        </div>
        <SubTabs
          tabs={TABS}
          active={tab}
          onChange={(key) => setTab(key as ProgressTab)}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === 'analytics' ? (
          <AnalyticsDashboard courseId={courseId ?? ''} userId={userId} />
        ) : (
          <PlannerPage />
        )}
      </div>
    </div>
  )
}
