// src/components/AnalyticsDashboard.tsx
import { useState, useEffect } from 'react'
import {
  TrendingUp,
  Target,
  Brain,
  Award,
  AlertCircle,
  BookOpen,
  Zap,
  BarChart3,
  Calendar,
  Flame
} from 'lucide-react'

interface AnalyticsData {
  topics_progress: Array<{
    topic: string
    mastery_level: number
    review_count: number
    last_reviewed: string
  }>
  study_streak: number
  weak_areas: string[]
  study_recommendations: string[]
  total_questions: number
  avg_confidence: number
  study_time_trend: Array<{
    date: string
    questions: number
  }>
}

interface AnalyticsDashboardProps {
  courseId: string
  userId: string
}

export default function AnalyticsDashboard({ courseId, userId }: AnalyticsDashboardProps) {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadAnalytics()
  }, [courseId, userId])

  const loadAnalytics = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/analytics/${courseId}/${userId}`)
      if (response.ok) {
        const data = await response.json()
        setAnalytics(data.analytics)
      }
    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  const getMasteryLabel = (level: number) => {
    if (level >= 0.9) return { label: 'Expert', color: 'text-emerald-400 bg-emerald-500/10' }
    if (level >= 0.8) return { label: 'Advanced', color: 'text-cyan-400 bg-cyan-500/10' }
    if (level >= 0.7) return { label: 'Proficient', color: 'text-amber-400 bg-amber-500/10' }
    if (level >= 0.5) return { label: 'Learning', color: 'text-amber-400 bg-amber-500/10' }
    return { label: 'Beginner', color: 'text-red-400 bg-red-500/10' }
  }

  const getMasteryWidth = (level: number) => `${Math.round(level * 100)}%`

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-zinc-800 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-zinc-800 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="text-center py-16">
          <BarChart3 className="w-16 h-16 text-zinc-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-zinc-400 mb-2">No Analytics Data Yet</h3>
          <p className="text-zinc-400">Start studying to see your progress!</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-50 flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-cyan-400" />
            Learning Analytics
          </h1>
          <p className="text-zinc-400 mt-2">Track your progress and identify areas for improvement</p>
        </div>

        <button
          onClick={loadAnalytics}
          className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700 transition-colors flex items-center gap-2"
        >
          <TrendingUp className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-500">Study Streak</p>
              <p className="text-2xl font-bold text-zinc-100">{analytics.study_streak}</p>
              <p className="text-sm text-zinc-500">days</p>
            </div>
            <div className="w-10 h-10 bg-zinc-800 rounded-lg flex items-center justify-center">
              <Flame className="w-5 h-5 text-zinc-400" />
            </div>
          </div>
        </div>

        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-500">Questions Asked</p>
              <p className="text-2xl font-bold text-zinc-100">{analytics.total_questions}</p>
              <p className="text-sm text-zinc-500">total</p>
            </div>
            <div className="w-10 h-10 bg-zinc-800 rounded-lg flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-zinc-400" />
            </div>
          </div>
        </div>

        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-500">Avg Confidence</p>
              <p className="text-2xl font-bold text-zinc-100">{Math.round(analytics.avg_confidence * 100)}%</p>
              <p className="text-sm text-zinc-500">score</p>
            </div>
            <div className="w-10 h-10 bg-zinc-800 rounded-lg flex items-center justify-center">
              <Target className="w-5 h-5 text-zinc-400" />
            </div>
          </div>
        </div>

        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-500">Topics Studied</p>
              <p className="text-2xl font-bold text-zinc-100">{analytics.topics_progress.length}</p>
              <p className="text-sm text-zinc-500">concepts</p>
            </div>
            <div className="w-10 h-10 bg-zinc-800 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-zinc-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Topic Progress */}
      <div className="bg-zinc-900/60 rounded-lg border border-zinc-800 p-6">
        <h2 className="text-2xl font-bold text-zinc-50 mb-6 flex items-center gap-3">
          <TrendingUp className="w-6 h-6 text-cyan-400" />
          Topic Mastery Progress
        </h2>

        {analytics.topics_progress.length === 0 ? (
          <div className="text-center py-8 text-zinc-400">
            <Brain className="w-12 h-12 text-zinc-400 mx-auto mb-3" />
            <p>No topics studied yet. Start asking questions to see progress!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {analytics.topics_progress.map((topic, index) => {
              const mastery = getMasteryLabel(topic.mastery_level)
              return (
                <div key={index} className="border border-zinc-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-zinc-50 capitalize">{topic.topic}</h3>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${mastery.color}`}>
                        {mastery.label}
                      </span>
                    </div>
                    <div className="text-sm text-zinc-400">
                      {topic.review_count} reviews
                    </div>
                  </div>

                  <div className="mb-2">
                    <div className="flex justify-between text-sm text-zinc-400 mb-1">
                      <span>Progress</span>
                      <span>{Math.round(topic.mastery_level * 100)}%</span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-2">
                      <div
                        className="bg-cyan-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: getMasteryWidth(topic.mastery_level) }}
                      ></div>
                    </div>
                  </div>

                  <div className="text-xs text-zinc-400">
                    Last reviewed: {new Date(topic.last_reviewed).toLocaleDateString()}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Weak Areas & Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weak Areas */}
        <div className="bg-zinc-900/60 rounded-lg border border-zinc-800 p-6">
          <h2 className="text-xl font-bold text-zinc-50 mb-4 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            Areas to Focus On
          </h2>

          {analytics.weak_areas.length === 0 ? (
            <div className="text-center py-8">
              <Award className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
              <p className="text-emerald-400 font-medium">Great job!</p>
              <p className="text-zinc-400 text-sm">No weak areas identified</p>
            </div>
          ) : (
            <div className="space-y-3">
              {analytics.weak_areas.map((area, index) => (
                <div key={index} className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="font-medium text-red-400 capitalize">{area}</span>
                  </div>
                  <p className="text-red-300 text-sm mt-1">
                    Consider reviewing this topic and practicing more problems
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Study Recommendations */}
        <div className="bg-zinc-900/60 rounded-lg border border-zinc-800 p-6">
          <h2 className="text-xl font-bold text-zinc-50 mb-4 flex items-center gap-3">
            <Zap className="w-5 h-5 text-amber-400" />
            Study Recommendations
          </h2>

          <div className="space-y-3">
            {analytics.study_recommendations.map((rec, index) => (
              <div key={index} className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-amber-400 mt-0.5" />
                  <p className="text-amber-200 text-sm">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Study Schedule Suggestion */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <Calendar className="w-6 h-6 text-cyan-400" />
          <h2 className="text-xl font-bold text-zinc-50">Suggested Study Schedule</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-zinc-900/60 rounded-lg p-4 border border-zinc-800">
            <h3 className="font-semibold text-zinc-200 mb-2">Today</h3>
            <p className="text-sm text-zinc-400">
              {analytics.weak_areas.length > 0
                ? `Review ${analytics.weak_areas[0]} concepts`
                : 'Great job! Try exploring new topics'
              }
            </p>
          </div>

          <div className="bg-zinc-900/60 rounded-lg p-4 border border-zinc-800">
            <h3 className="font-semibold text-zinc-200 mb-2">This Week</h3>
            <p className="text-sm text-zinc-400">
              Practice problems on your strongest topics to maintain mastery
            </p>
          </div>

          <div className="bg-zinc-900/60 rounded-lg p-4 border border-zinc-800">
            <h3 className="font-semibold text-zinc-200 mb-2">Next Steps</h3>
            <p className="text-sm text-zinc-400">
              {analytics.total_questions < 50
                ? 'Ask more questions to get better insights'
                : 'Consider taking a practice exam'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
