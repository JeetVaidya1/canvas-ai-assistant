/**
 * Tiny client-side fuzzy matcher for the command palette.
 * Returns 0 for no match; higher is better. Substring beats subsequence,
 * word-start hits beat mid-word hits, shorter targets beat longer ones.
 */
export function fuzzyScore(query: string, target: string): number {
  const q = query.trim().toLowerCase()
  const t = target.toLowerCase()
  if (!q) return 1
  if (q === t) return 1000

  const substringAt = t.indexOf(q)
  if (substringAt >= 0) {
    const wordStart = substringAt === 0 || /[\s›/-]/.test(t[substringAt - 1])
    return 500 + (wordStart ? 100 : 0) - substringAt - t.length * 0.1
  }

  // Subsequence match: every query char in order, scoring word-start hits.
  let score = 0
  let ti = 0
  for (const ch of q) {
    let found = -1
    for (let i = ti; i < t.length; i++) {
      if (t[i] === ch) {
        found = i
        break
      }
    }
    if (found === -1) return 0
    const wordStart = found === 0 || /[\s›/-]/.test(t[found - 1])
    score += wordStart ? 10 : 1
    // Contiguity bonus
    if (found === ti) score += 4
    ti = found + 1
  }
  return Math.max(1, score - t.length * 0.1)
}
