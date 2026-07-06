import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Zap, Target } from 'lucide-react'
import { SubTabs } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'
import { trackVisit } from '@/hooks/useRecentActivity'
import QuizMode from '@/components/QuizMode'
import PracticeMode from '@/components/PracticeMode'
import type { ModeChangeHandler } from '@/components/practice/types'

type Mode = 'quiz' | 'problems'

/** Practice destination — merges the old "Quiz" and "Practice" tools (which were
 *  the same idea) into one self-test surface with a mode toggle. */
export default function PracticePage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('quiz')

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'practice')
  }, [courseId])

  // Results-screen next actions land here: same-page mode switches stay local,
  // everything else routes to the consolidated destinations.
  const handleModeChange = useCallback<ModeChangeHandler>(
    (next) => {
      if (next === 'quiz') setMode('quiz')
      else if (next === 'practice') setMode('problems')
      else if (!courseId) return
      else if (next === 'analytics') void navigate(`/course/${courseId}/progress`)
      else if (next === 'chat') void navigate(`/course/${courseId}/learn`)
      else void navigate(`/course/${courseId}/kit`)
    },
    [courseId, navigate],
  )

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 h-14 border-b border-[#18181d] flex items-center justify-between flex-shrink-0 gap-4">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 pl-1.5">
          {mode === 'quiz' ? 'Drill fast, learn faster' : 'Work it through, deeply'}
        </span>
        <SubTabs
          tabs={[
            { key: 'quiz', label: 'Quick Quiz', icon: <Zap className="w-4 h-4" />, hint: 'Rapid multiple-choice, graded instantly' },
            { key: 'problems', label: 'Problem Set', icon: <Target className="w-4 h-4" />, hint: 'Open-ended, adaptive difficulty' },
          ]}
          active={mode}
          onChange={(k) => setMode(k as Mode)}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {mode === 'quiz'
          ? <QuizMode courseId={courseId ?? ''} userId={userId} onModeChange={handleModeChange} />
          : <PracticeMode courseId={courseId ?? ''} userId={userId} onModeChange={handleModeChange} />}
      </div>
    </div>
  )
}
