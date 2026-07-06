// src/components/QuizAsisstant.tsx (filename typo is intentional — do not rename)
import React, { useState, useRef, useEffect } from 'react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Textarea } from '@/components/ui/Input'
import { ErrorState } from '@/components/ui/States'
import {
  Brain,
  Copy,
  AlertCircle,
  Clock,
  Lightbulb,
  Target,
  Zap,
  HelpCircle,
  FileText,
  TrendingUp,
  RotateCcw,
  Plus,
  User,
  Globe
} from 'lucide-react'
import { assistWithQuiz, type QuizResponse } from '../lib/api'

interface QuizAssistantProps {
  courseId: string
  sessionId?: string
  onQuizSubmit?: (question: string, response: QuizResponse) => void
}

interface QuizConversation {
  id: string
  question: string
  response: QuizResponse
  timestamp: string
}

type BadgeTone = 'success' | 'warning' | 'danger'

export default function QuizAssistant({ courseId, sessionId, onQuizSubmit }: QuizAssistantProps) {
  const [question, setQuestion] = useState('')
  const [conversations, setConversations] = useState<QuizConversation[]>([])
  const [loading, setLoading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'
    }
  }, [question])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || !courseId) return

    const currentQuestion = question.trim()
    setQuestion('')
    setLoading(true)

    try {
      const data = await assistWithQuiz(currentQuestion, courseId, sessionId)

      const newConversation: QuizConversation = {
        id: Date.now().toString(),
        question: currentQuestion,
        response: data,
        timestamp: new Date().toISOString()
      }

      setConversations(prev => [...prev, newConversation])

      if (onQuizSubmit) {
        onQuizSubmit(currentQuestion, data)
      }

    } catch (error) {
      const errorResponse: QuizResponse = {
        status: 'error',
        answer: 'Sorry, I encountered an error processing your question.',
        explanation: 'Please try again or rephrase your question.',
        confidence: 0,
        question_type: 'unknown',
        study_tips: [],
        similar_concepts: [],
        estimated_time: '',
        relevant_sources: [],
        error: error instanceof Error ? error.message : 'Unknown error'
      }

      const errorConversation: QuizConversation = {
        id: Date.now().toString(),
        question: currentQuestion,
        response: errorResponse,
        timestamp: new Date().toISOString()
      }

      setConversations(prev => [...prev, errorConversation])
    } finally {
      setLoading(false)
    }
  }

  const clearConversations = () => {
    setConversations([])
    setQuestion('')
  }

  const copyAnswer = (answer: string) => {
    void navigator.clipboard.writeText(answer)
  }

  const getConfidenceTone = (confidence: number): BadgeTone => {
    if (confidence >= 0.8) return 'success'
    if (confidence >= 0.6) return 'warning'
    return 'danger'
  }

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.9) return 'Very High'
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  const formatQuestionType = (type: string) => {
    return type.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  const getSourceIcon = (source: string) => {
    if (source.toLowerCase().includes('web') || source.toLowerCase().includes('search')) {
      return <Globe className="w-3 h-3 text-cyan-400" />
    }
    return <FileText className="w-3 h-3 text-zinc-400" />
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="card-surface accent-top rounded-t-xl p-5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0">
              <Brain className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-0.5">Answer Helper</p>
              <h2 className="text-lg font-semibold text-zinc-100">Quiz Assistant</h2>
              <p className="text-sm text-zinc-400">
                Paste any quiz question and get intelligent help with explanations
              </p>
            </div>
          </div>

          {conversations.length > 0 && (
            <Button
              variant="secondary"
              onClick={clearConversations}
              leftIcon={<RotateCcw className="w-4 h-4" />}
            >
              Clear All
            </Button>
          )}
        </div>
      </div>

      {/* Conversation History */}
      {conversations.length > 0 && (
        <div className="card-surface border-t-0 rounded-none max-h-96 overflow-y-auto">
          <div className="p-6 space-y-6">
            {conversations.map((conv) => (
              <div key={conv.id} className="space-y-4">
                {/* User Question */}
                <div className="flex items-start gap-4 flex-row-reverse">
                  <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-zinc-300" />
                  </div>
                  <div className="max-w-3xl text-right">
                    <div className="inline-block px-4 py-3 rounded-lg bg-gradient-brand text-white glow-brand-sm">
                      <p className="whitespace-pre-wrap leading-relaxed">
                        {conv.question}
                      </p>
                    </div>
                    <p className="text-xs text-zinc-400 mt-2 px-1">
                      {new Date(conv.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </p>
                  </div>
                </div>

                {/* AI Response */}
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0">
                    <Brain className="w-4 h-4 text-cyan-300" />
                  </div>
                  <div className="flex-1">
                    {conv.response.status === 'success' ? (
                      <div className="space-y-4">
                        {/* Answer Header */}
                        <div className="flex items-center gap-3">
                          <Target className="w-5 h-5 text-cyan-300" />
                          <span className="font-semibold text-zinc-50">Answer</span>
                          <Badge tone={getConfidenceTone(conv.response.confidence)}>
                            {getConfidenceText(conv.response.confidence)}
                          </Badge>
                          <Badge tone="accent">
                            {formatQuestionType(conv.response.question_type)}
                          </Badge>
                        </div>

                        {/* Main Answer */}
                        <div className="bg-gradient-brand-soft border border-cyan-400/15 rounded-lg p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                              <div className="text-lg font-semibold text-zinc-100">
                                <Markdown content={conv.response.answer} />
                              </div>
                            </div>
                            <button
                              onClick={() => copyAnswer(conv.response.answer)}
                              className="p-2 text-cyan-300 hover:bg-cyan-400/10 rounded-lg transition-colors flex-shrink-0"
                              aria-label="Copy answer"
                              title="Copy answer"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                          </div>
                        </div>

                        {/* Explanation */}
                        <div>
                          <div className="flex items-center gap-2 mb-3">
                            <Lightbulb className="w-5 h-5 text-amber-400" />
                            <span className="font-semibold text-zinc-50">Explanation</span>
                          </div>
                          <div className="bg-white/[0.04] border border-white/10 rounded-lg p-4">
                            <Markdown content={conv.response.explanation} className="text-zinc-300" />
                          </div>
                        </div>

                        {/* Quick Stats */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="bg-white/[0.04] border border-white/10 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                              <Clock className="w-4 h-4 text-zinc-400" />
                              <div>
                                <div className="text-xs text-zinc-500 font-medium">Est. Time</div>
                                <div className="text-sm font-semibold text-zinc-100">{conv.response.estimated_time}</div>
                              </div>
                            </div>
                          </div>

                          <div className="bg-white/[0.04] border border-white/10 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                              <FileText className="w-4 h-4 text-zinc-400" />
                              <div>
                                <div className="text-xs text-zinc-500 font-medium">Sources</div>
                                <div className="text-sm font-semibold text-zinc-100">{conv.response.relevant_sources.length}</div>
                              </div>
                            </div>
                          </div>

                          <div className="bg-white/[0.04] border border-white/10 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                              <TrendingUp className="w-4 h-4 text-zinc-400" />
                              <div>
                                <div className="text-xs text-zinc-500 font-medium">Confidence</div>
                                <div className="text-sm font-semibold text-zinc-100">
                                  {Math.round(conv.response.confidence * 100)}%
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Study Tips */}
                        {conv.response.study_tips.length > 0 && (
                          <div>
                            <div className="flex items-center gap-2 mb-3">
                              <Lightbulb className="w-4 h-4 text-amber-400" />
                              <span className="font-semibold text-zinc-50">Study Tips</span>
                            </div>
                            <div className="space-y-2">
                              {conv.response.study_tips.map((tip, index) => (
                                <div key={index} className="bg-amber-500/10 border border-amber-400/25 rounded-lg p-3">
                                  <p className="text-sm text-amber-300">{tip}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Sources */}
                        {conv.response.relevant_sources.length > 0 && (
                          <div>
                            <div className="flex items-center gap-2 mb-3">
                              <FileText className="w-4 h-4 text-zinc-400" />
                              <span className="font-semibold text-zinc-50">Sources Referenced</span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {conv.response.relevant_sources.map((source, index) => (
                                <Badge key={index} tone="neutral" icon={getSourceIcon(source)}>
                                  {source}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      /* Error State */
                      <ErrorState
                        compact
                        title={conv.response.explanation || 'Unable to process this question — try rephrasing it.'}
                      />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className={`card-surface border-t-0 p-6 ${conversations.length === 0 ? 'rounded-none' : ''}`}>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            ref={textareaRef}
            label={conversations.length > 0 ? 'Ask another question:' : 'Paste your quiz question here:'}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Example:&#10;&#10;Which of the following best describes photosynthesis?&#10;A) The process by which plants break down glucose&#10;B) The process by which plants convert light energy into chemical energy&#10;C) The process by which plants absorb water&#10;D) The process by which plants release oxygen"
            className="min-h-[120px] p-4 resize-none"
            disabled={loading}
          />

          <div className="flex items-center gap-3">
            <Button
              type="submit"
              size="lg"
              disabled={loading || !question.trim() || !courseId}
              loading={loading}
              leftIcon={<Zap className="w-5 h-5" />}
            >
              {loading ? 'Analyzing...' : conversations.length > 0 ? 'Ask Another' : 'Get Answer'}
            </Button>

            {conversations.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                size="lg"
                onClick={() => setQuestion('')}
                leftIcon={<Plus className="w-4 h-4 rotate-45" />}
              >
                Clear
              </Button>
            )}

            {!courseId && (
              <div className="flex items-center gap-2 text-amber-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                Select a course first
              </div>
            )}
          </div>
        </form>
      </div>

      {/* Footer Tips */}
      {conversations.length === 0 && (
        <div className="card-surface border-t-0 rounded-t-none rounded-b-xl p-6">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0">
              <HelpCircle className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <h4 className="font-semibold text-zinc-50 mb-2">Pro Tips for Best Results:</h4>
              <ul className="text-sm text-zinc-400 space-y-1">
                <li>Copy the entire question including all answer choices</li>
                <li>Include any context or background information provided</li>
                <li>For math problems, include diagrams or formulas if mentioned</li>
                <li>The more complete the question, the better the explanation</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
