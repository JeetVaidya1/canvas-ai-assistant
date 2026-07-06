import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { BarChart3, Calendar } from 'lucide-react'
import AnalyticsDashboard from '@/components/AnalyticsDashboard'
import PlannerPage from '@/pages/PlannerPage'
import { SubTabs, type SubTab } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'

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
  const [tab, setTab] = useState<ProgressTab>('analytics')

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 h-14 border-b border-[#18181d] flex items-center justify-between flex-shrink-0 gap-4">
        <span className="text-xs text-zinc-500 pl-1.5">
          {tab === 'analytics' ? 'See where you stand' : 'Plan your path to the exam'}
        </span>
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
