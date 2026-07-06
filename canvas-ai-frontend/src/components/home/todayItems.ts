import type { Readiness } from '@/lib/api'
import type { CourseTopic } from '@/lib/api/topics'

/** One row of the Today checklist. */
export interface TodayItem {
  key: string
  label: string
  /** Secondary line under the label (topic name, destination, …). */
  detail?: string
  /** Honest, rough time estimate in minutes. */
  etaMin: number
  /** Course-relative destination segment, e.g. 'practice'. */
  to: string
  /** Onboarding steps already completed render checked. */
  done?: boolean
}

export interface TodayPlan {
  items: TodayItem[]
  onboarding: boolean
  /** Sum of estimates for the remaining (not-done) items. */
  totalMin: number
}

const PAGE_LABELS: Record<string, string> = {
  learn: 'Learn',
  chat: 'Learn',
  practice: 'Practice',
  quiz: 'Practice',
  exam: 'Exam',
  exams: 'Exam',
  kit: 'Study Kit',
  notes: 'Study Kit',
  progress: 'Progress',
}

export function pageLabel(page: string): string {
  return PAGE_LABELS[page] ?? page
}

function normalizeTopic(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '')
}

/** Contains-match between a Course Brain topic name and a mastery label. */
export function topicsMatch(a: string, b: string): boolean {
  const na = normalizeTopic(a)
  const nb = normalizeTopic(b)
  if (!na || !nb) return false
  return na.includes(nb) || nb.includes(na)
}

/**
 * Mastery pct for a Course Brain topic name from readiness.by_topic.
 * Returns null when no measured entry matches — callers render no bar
 * rather than a fake one.
 */
export function masteryForTopic(name: string, readiness: Readiness | null): number | null {
  if (!readiness) return null
  const hit = readiness.by_topic.find((t) => t.has_data && topicsMatch(name, t.topic))
  return hit ? Math.round(hit.mastery_pct) : null
}

/** Weakest topic: lowest measured mastery, else first readiness gap, else first Course Brain topic. */
export function weakestTopic(
  readiness: Readiness | null,
  topics: readonly CourseTopic[] | undefined,
): string | null {
  const measured = (readiness?.by_topic ?? []).filter((t) => t.has_data)
  if (measured.length > 0) {
    const weakest = measured.reduce((min, t) => (t.mastery_pct < min.mastery_pct ? t : min))
    return weakest.topic
  }
  if (readiness && readiness.gaps.length > 0) return readiness.gaps[0]
  return topics?.[0]?.name ?? null
}

/**
 * Assemble the Today checklist client-side from whatever data resolved.
 * When there is no study signal at all, falls back to an honest onboarding
 * checklist (materials → first question → first quiz).
 */
export function buildTodayPlan(input: {
  dueCount: number | null
  readiness: Readiness | null
  topics: readonly CourseTopic[] | undefined
  recentPage: string | null
  hasFiles: boolean
}): TodayPlan {
  const { dueCount, readiness, topics, recentPage, hasFiles } = input
  const items: TodayItem[] = []

  if (dueCount !== null && dueCount > 0) {
    items.push({
      key: 'review',
      label: `Review ${dueCount} due card${dueCount === 1 ? '' : 's'}`,
      detail: 'Spaced repetition — clear them before they fade',
      etaMin: Math.max(2, Math.ceil(dueCount / 2)),
      to: 'progress',
    })
  }

  const weakest = weakestTopic(readiness, topics)
  if (weakest) {
    items.push({
      key: 'drill',
      label: 'Drill your weakest topic',
      detail: weakest,
      etaMin: 10,
      to: 'practice',
    })
  }

  if (recentPage) {
    items.push({
      key: 'continue',
      label: 'Continue where you left off',
      detail: pageLabel(recentPage),
      etaMin: 5,
      to: recentPage,
    })
  }

  if (items.length > 0) {
    return { items, onboarding: false, totalMin: items.reduce((sum, i) => sum + i.etaMin, 0) }
  }

  const onboardingItems: TodayItem[] = [
    {
      key: 'materials',
      label: 'Add your course materials',
      detail: 'PDFs, slides & docs power every tool',
      etaMin: 2,
      to: 'materials',
      done: hasFiles,
    },
    {
      key: 'ask',
      label: 'Ask your first question',
      detail: 'Grounded answers with page citations',
      etaMin: 5,
      to: 'learn',
    },
    {
      key: 'first-quiz',
      label: 'Take a first quiz',
      detail: 'Starts your mastery tracking',
      etaMin: 5,
      to: 'practice',
    },
  ]
  return {
    items: onboardingItems,
    onboarding: true,
    totalMin: onboardingItems.filter((i) => !i.done).reduce((sum, i) => sum + i.etaMin, 0),
  }
}
