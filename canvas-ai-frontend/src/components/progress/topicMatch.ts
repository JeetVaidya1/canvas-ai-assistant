import type { CourseTopic } from '@/lib/api/topics'
import type { Readiness } from '@/lib/api'

export type AnalyticsTopicRow = Readiness['by_topic'][number]

export interface MatchedTopicRow {
  slug: string
  name: string
  /** Weighted mastery across matched analytics rows — null = not practiced yet. */
  masteryPct: number | null
}

export interface TopicMasteryMatch {
  /** One row per Course Brain topic, in course order. */
  matched: MatchedTopicRow[]
  /** Analytics rows with data that matched no course topic — never hide data. */
  unmatched: AnalyticsTopicRow[]
}

/** Lowercase, strip punctuation, collapse whitespace — the comparison key. */
export function normalizeTopicKey(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
}

function isContainsMatch(a: string, b: string): boolean {
  if (a.length === 0 || b.length === 0) return false
  return a.includes(b) || b.includes(a)
}

/** Weight-aware average; falls back to a plain mean when weights are zero. */
function weightedMastery(rows: readonly AnalyticsTopicRow[]): number {
  const totalWeight = rows.reduce((sum, r) => sum + (r.weight > 0 ? r.weight : 0), 0)
  if (totalWeight > 0) {
    return rows.reduce((sum, r) => sum + r.mastery_pct * (r.weight > 0 ? r.weight : 0), 0) / totalWeight
  }
  return rows.reduce((sum, r) => sum + r.mastery_pct, 0) / rows.length
}

/**
 * Re-key analytics mastery onto the Course Brain taxonomy: each analytics
 * by_topic row is matched to a course topic via normalized contains-match
 * (either direction), so legacy strings like "301 3 Excel" land under the
 * clean topic name. Rows that match nothing are returned as `unmatched`.
 */
export function matchAnalyticsToTopics(
  topics: readonly CourseTopic[],
  byTopic: readonly AnalyticsTopicRow[],
): TopicMasteryMatch {
  const ordered = [...topics].sort((a, b) => a.position - b.position)
  const keys = ordered.map((t) => normalizeTopicKey(t.name))
  const dataRows = byTopic.filter((r) => r.has_data)

  const rowsByTopic = new Map<string, AnalyticsTopicRow[]>()
  const unmatched: AnalyticsTopicRow[] = []

  for (const row of dataRows) {
    const rowKey = normalizeTopicKey(row.topic)
    const idx = keys.findIndex((k) => isContainsMatch(k, rowKey))
    if (idx === -1) {
      unmatched.push(row)
      continue
    }
    const slug = ordered[idx].slug
    rowsByTopic.set(slug, [...(rowsByTopic.get(slug) ?? []), row])
  }

  const matched = ordered.map((t) => {
    const rows = rowsByTopic.get(t.slug)
    return {
      slug: t.slug,
      name: t.name,
      masteryPct: rows && rows.length > 0 ? weightedMastery(rows) : null,
    }
  })

  return { matched, unmatched }
}
