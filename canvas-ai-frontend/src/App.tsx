import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate, useParams } from 'react-router-dom'
import AppLayout from '@/components/layout/AppLayout'
import RequireAuth from '@/components/RequireAuth'
import ErrorBoundary from '@/components/shared/ErrorBoundary'
import LoadingSpinner from '@/components/shared/LoadingSpinner'

const LandingPage = lazy(() => import('@/pages/LandingPage'))
const PricingPage = lazy(() => import('@/pages/PricingPage'))
const TermsPage = lazy(() => import('@/pages/TermsPage'))
const PrivacyPage = lazy(() => import('@/pages/PrivacyPage'))
const HelpPage = lazy(() => import('@/pages/HelpPage'))
const LoginPage = lazy(() => import('@/pages/LoginPage'))
const Dashboard = lazy(() => import('@/pages/Dashboard'))
const CourseHome = lazy(() => import('@/pages/CourseHome'))
const CourseOverview = lazy(() => import('@/pages/CourseOverview'))
const LearnPage = lazy(() => import('@/pages/LearnPage'))
const PracticePage = lazy(() => import('@/pages/PracticePage'))
const ExamsPage = lazy(() => import('@/pages/ExamsPage'))
const StudyKitPage = lazy(() => import('@/pages/StudyKitPage'))
const ProgressPage = lazy(() => import('@/pages/ProgressPage'))
const SettingsPage = lazy(() => import('@/pages/SettingsPage'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <LoadingSpinner size="lg" />
    </div>
  )
}

/** Redirect old tool paths to the new consolidated destinations. */
function RedirectCourse({ to }: { to: string }) {
  const { courseId } = useParams<{ courseId: string }>()
  return <Navigate to={`/course/${courseId}/${to}`} replace />
}

function Page({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<PageFallback />}>
      <ErrorBoundary>{children}</ErrorBoundary>
    </Suspense>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <div className="relative z-10">
      <Routes>
        <Route path="/" element={<Suspense fallback={<PageFallback />}><LandingPage /></Suspense>} />
        <Route path="/pricing" element={<Suspense fallback={<PageFallback />}><PricingPage /></Suspense>} />
        <Route path="/terms" element={<Suspense fallback={<PageFallback />}><TermsPage /></Suspense>} />
        <Route path="/privacy" element={<Suspense fallback={<PageFallback />}><PrivacyPage /></Suspense>} />
        <Route path="/help" element={<Suspense fallback={<PageFallback />}><HelpPage /></Suspense>} />
        <Route path="/login" element={<Suspense fallback={<PageFallback />}><LoginPage /></Suspense>} />
        <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route path="dashboard" element={<Page><Dashboard /></Page>} />

          {/* Six intent-based destinations */}
          <Route path="course/:courseId" element={<Page><CourseHome /></Page>} />
          <Route path="course/:courseId/materials" element={<Page><CourseOverview /></Page>} />
          <Route path="course/:courseId/learn" element={<Page><LearnPage /></Page>} />
          <Route path="course/:courseId/practice" element={<Page><PracticePage /></Page>} />
          <Route path="course/:courseId/exam" element={<Page><ExamsPage /></Page>} />
          <Route path="course/:courseId/kit" element={<Page><StudyKitPage /></Page>} />
          <Route path="course/:courseId/progress" element={<Page><ProgressPage /></Page>} />

          {/* Old tool paths → new destinations (keeps links + recent-activity working) */}
          <Route path="course/:courseId/chat" element={<RedirectCourse to="learn" />} />
          <Route path="course/:courseId/tutor" element={<RedirectCourse to="learn" />} />
          <Route path="course/:courseId/quiz" element={<RedirectCourse to="practice" />} />
          <Route path="course/:courseId/notes" element={<RedirectCourse to="kit" />} />
          <Route path="course/:courseId/audio" element={<RedirectCourse to="kit" />} />
          <Route path="course/:courseId/analytics" element={<RedirectCourse to="progress" />} />
          <Route path="course/:courseId/planner" element={<RedirectCourse to="progress" />} />
          <Route path="course/:courseId/exams" element={<RedirectCourse to="exam" />} />

          <Route path="settings" element={<Page><SettingsPage /></Page>} />
          <Route path="*" element={<Page><NotFoundPage /></Page>} />
        </Route>
        </Route>
      </Routes>
      </div>
    </ErrorBoundary>
  )
}
