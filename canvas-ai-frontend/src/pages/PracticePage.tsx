import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { Brain, Target } from 'lucide-react'
import { SubTabs } from '@/components/ui/SubTabs'
import { useUser } from '@/hooks/useUser'
import { useCourses } from '@/hooks/useCourses'
import { trackVisit } from '@/hooks/useRecentActivity'
import QuizMode from '@/components/QuizMode'
import PracticeMode from '@/components/PracticeMode'

type Mode = 'quiz' | 'problems'

/** Practice destination — merges the old "Quiz" and "Practice" tools (which were
 *  the same idea) into one self-test surface with a mode toggle. */
export default function PracticePage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const { data: courses } = useCourses()
  const course = courses?.find((c) => c.course_id === courseId)
  const [mode, setMode] = useState<Mode>('quiz')

  useEffect(() => {
    if (courseId) trackVisit(courseId, 'practice')
  }, [courseId])

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-5 pb-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0 gap-4">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand">Practice</p>
          <h1 className="text-lg font-semibold text-zinc-100 truncate">{course?.title ?? 'Practice'}</h1>
        </div>
        <SubTabs
          tabs={[
            { key: 'quiz', label: 'Quick Quiz', icon: <Brain className="w-4 h-4" />, hint: 'Multiple-choice, graded instantly' },
            { key: 'problems', label: 'Problem Set', icon: <Target className="w-4 h-4" />, hint: 'Open-ended, adaptive difficulty' },
          ]}
          active={mode}
          onChange={(k) => setMode(k as Mode)}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {mode === 'quiz'
          ? <QuizMode courseId={courseId ?? ''} userId={userId} />
          : <PracticeMode courseId={courseId ?? ''} userId={userId} />}
      </div>
    </div>
  )
}
