/** Maps a 0-100 score to the app-wide semantic tone. Single source of truth
 *  for how readiness/mastery scores are colored and labelled. */
export function scoreTone(score: number): { stroke: string; text: string; label: string } {
  if (score >= 70) return { stroke: '#34d399', text: 'text-emerald-300', label: 'On track' }
  if (score >= 40) return { stroke: '#fbbf24', text: 'text-amber-300', label: 'Getting there' }
  return { stroke: '#fb7185', text: 'text-rose-300', label: 'Needs work' }
}
