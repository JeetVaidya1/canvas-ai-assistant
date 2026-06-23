import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import AppLayout from '@/components/layout/AppLayout'
import ErrorBoundary from '@/components/shared/ErrorBoundary'
import LoadingSpinner from '@/components/shared/LoadingSpinner'
import { ShaderCanvas } from '@/components/ui/animated-shader-hero'
import { FallingBooks } from '@/components/ui/falling-books'

const LandingPage = lazy(() => import('@/pages/LandingPage'))
const Dashboard = lazy(() => import('@/pages/Dashboard'))
const CourseOverview = lazy(() => import('@/pages/CourseOverview'))
const ChatPage = lazy(() => import('@/pages/ChatPage'))
const TutorPage = lazy(() => import('@/pages/TutorPage'))
const QuizPage = lazy(() => import('@/pages/QuizPage'))
const PracticePage = lazy(() => import('@/pages/PracticePage'))
const NotesPage = lazy(() => import('@/pages/NotesPage'))
const ExamsPage = lazy(() => import('@/pages/ExamsPage'))
const AnalyticsPage = lazy(() => import('@/pages/AnalyticsPage'))
const PlannerPage = lazy(() => import('@/pages/PlannerPage'))
const AudioPage = lazy(() => import('@/pages/AudioPage'))
const SettingsPage = lazy(() => import('@/pages/SettingsPage'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <LoadingSpinner size="lg" />
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <ShaderCanvas />
      <FallingBooks />
      <div className="relative z-10">
      <Routes>
        <Route path="/" element={<Suspense fallback={<PageFallback />}><LandingPage /></Suspense>} />
        <Route element={<AppLayout />}>
          <Route path="dashboard" element={<Suspense fallback={<PageFallback />}><Dashboard /></Suspense>} />
          <Route path="course/:courseId" element={<Suspense fallback={<PageFallback />}><CourseOverview /></Suspense>} />
          <Route path="course/:courseId/chat" element={<Suspense fallback={<PageFallback />}><ChatPage /></Suspense>} />
          <Route path="course/:courseId/tutor" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><TutorPage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/quiz" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><QuizPage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/practice" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><PracticePage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/notes" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><NotesPage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/exams" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><ExamsPage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/analytics" element={<Suspense fallback={<PageFallback />}><ErrorBoundary><AnalyticsPage /></ErrorBoundary></Suspense>} />
          <Route path="course/:courseId/planner" element={<Suspense fallback={<PageFallback />}><PlannerPage /></Suspense>} />
          <Route path="course/:courseId/audio" element={<Suspense fallback={<PageFallback />}><AudioPage /></Suspense>} />
          <Route path="settings" element={<Suspense fallback={<PageFallback />}><SettingsPage /></Suspense>} />
          <Route path="*" element={<Suspense fallback={<PageFallback />}><NotFoundPage /></Suspense>} />
        </Route>
      </Routes>
      </div>
    </ErrorBoundary>
  )
}
