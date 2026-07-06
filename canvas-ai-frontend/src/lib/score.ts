/** Maps a 0-100 score to the app-wide semantic tone. Single source of truth
 *  for how readiness/mastery scores are colored and labelled (Paper & Ink:
 *  muted, ink-like semantics — never neon). */
export function scoreTone(score: number): { stroke: string; text: string; label: string } {
  if (score >= 70) return { stroke: '#2f7d5c', text: 'text-success', label: 'On track' }
  if (score >= 40) return { stroke: '#a8741a', text: 'text-warning', label: 'Getting there' }
  return { stroke: '#bb4444', text: 'text-danger', label: 'Needs work' }
}
