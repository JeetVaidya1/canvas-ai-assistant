// Exam destination — thin composition over the exam state machine.
//
// All state (setup choices, live session, countdown + resume-on-refresh
// persistence, grading) lives in useExamSession; each phase renders one
// focused screen from src/components/exam/:
//
//   setup    → ExamSetup      (difficulty/count picker + past-paper upload)
//   preStart → ExamPreStart   (confirmation / resume)
//   live     → ExamLive       (sticky timer bar, question column, navigator)
//   results  → ExamResults    (score ring, AI-judge verdicts, concept breakdown)
import { useExamSession } from './exam/useExamSession'
import { ExamSetup } from './exam/ExamSetup'
import { ExamPreStart } from './exam/ExamPreStart'
import { ExamLive } from './exam/ExamLive'
import { ExamResults } from './exam/ExamResults'

interface ExamModeProps {
  courseId: string
  userId: string
}

export default function ExamMode({ courseId, userId }: ExamModeProps) {
  const exam = useExamSession(courseId, userId)

  switch (exam.phase) {
    case 'results':
      // examResults is guaranteed non-null in this phase by the state machine.
      return exam.examResults ? <ExamResults results={exam.examResults} onNewExam={exam.resetExam} /> : null

    case 'setup':
      return (
        <ExamSetup
          difficulty={exam.examDifficulty}
          onDifficultyChange={exam.setExamDifficulty}
          questionCount={exam.examQuestionCount}
          onQuestionCountChange={exam.setExamQuestionCount}
          loading={exam.loading}
          genError={exam.genError}
          canGenerate={!!courseId}
          onGenerate={exam.generateExam}
          onLoadSample={exam.loadSample}
          uploading={exam.uploading}
          analysisSummary={exam.analysisSummary}
          onUploadPaper={exam.uploadPaper}
        />
      )

    case 'preStart':
      // examSession is guaranteed non-null in this phase by the state machine.
      return exam.examSession ? (
        <ExamPreStart
          session={exam.examSession}
          timeRemaining={exam.timeRemaining}
          onStart={exam.startExam}
          onAbandon={exam.abandonExam}
        />
      ) : null

    case 'live':
      return <ExamLive exam={exam} />
  }
}
