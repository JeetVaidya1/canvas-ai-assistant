import { useCallback, useEffect, useState } from 'react'
import { Button } from '@/components/ui/Button'

type TourPage = 'dashboard' | 'courseHome'

interface TourState {
  readonly step: number
  readonly done: boolean
}

interface TourStep {
  readonly page: TourPage
  readonly anchor: string
  readonly title: string
  readonly body: string
}

const TOUR_KEY = 'vindexa_tour_v1'
const CARD_WIDTH = 288
const CARD_EST_HEIGHT = 150
const ANCHOR_GAP = 10
const VIEWPORT_EDGE = 8

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

const STEPS: readonly TourStep[] = [
  {
    page: 'dashboard',
    anchor: 'course-switcher',
    title: 'Courses live up here',
    body: 'Switch between your courses or create a new one from the top bar — it follows you everywhere.',
  },
  {
    page: 'dashboard',
    anchor: 'command-k',
    title: 'Jump anywhere',
    body: `Press ${IS_MAC ? '⌘K' : 'Ctrl+K'} to jump anywhere or start an action.`,
  },
  {
    page: 'courseHome',
    anchor: 'today-panel',
    title: 'Start with Today',
    body: 'This checklist turns due reviews and weak topics into a short, concrete plan for each course.',
  },
]

function readTourState(): TourState {
  try {
    const raw = localStorage.getItem(TOUR_KEY)
    if (!raw) return { step: 0, done: false }
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null) return { step: 0, done: false }
    const candidate = parsed as Partial<TourState>
    return {
      step: typeof candidate.step === 'number' && candidate.step >= 0 ? candidate.step : 0,
      done: candidate.done === true,
    }
  } catch {
    return { step: 0, done: false }
  }
}

function writeTourState(state: TourState): void {
  try {
    localStorage.setItem(TOUR_KEY, JSON.stringify(state))
  } catch {
    // Storage blocked/full — the tour just re-offers next visit.
  }
}

interface CalloutPosition {
  readonly top: number
  readonly left: number
  readonly caretLeft: number
  readonly placement: 'below' | 'above'
}

/** Fixed-position placement near the anchor, clamped to the viewport. */
function computePosition(rect: DOMRect): CalloutPosition {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const left = Math.min(
    Math.max(rect.left + rect.width / 2 - CARD_WIDTH / 2, VIEWPORT_EDGE),
    Math.max(VIEWPORT_EDGE, vw - CARD_WIDTH - VIEWPORT_EDGE),
  )
  const caretLeft = Math.min(Math.max(rect.left + rect.width / 2 - left - 6, 14), CARD_WIDTH - 26)
  const fitsBelow = rect.bottom + ANCHOR_GAP + CARD_EST_HEIGHT <= vh - VIEWPORT_EDGE
  const fitsAbove = rect.top - ANCHOR_GAP - CARD_EST_HEIGHT >= VIEWPORT_EDGE
  if (fitsBelow || !fitsAbove) {
    const top = Math.min(rect.bottom + ANCHOR_GAP, vh - CARD_EST_HEIGHT - VIEWPORT_EDGE)
    return { top, left, caretLeft, placement: 'below' }
  }
  return { top: rect.top - ANCHOR_GAP - CARD_EST_HEIGHT, left, caretLeft, placement: 'above' }
}

interface CoachMarksProps {
  page: TourPage
}

/**
 * First-run coach marks. Steps 1-2 anchor to the top bar on the Dashboard;
 * step 3 anchors to the Today panel the first time the user reaches a course
 * home afterwards. Progress persists in localStorage; renders nothing once
 * the tour is done or skipped. Missing anchors are skipped gracefully.
 */
export function CoachMarks({ page }: CoachMarksProps) {
  const [tour, setTour] = useState<TourState>(readTourState)
  const [pos, setPos] = useState<CalloutPosition | null>(null)

  const step = tour.done ? undefined : STEPS[tour.step]
  const active = step && step.page === page ? step : undefined

  const advance = useCallback(() => {
    const nextStep = tour.step + 1
    const next: TourState = { step: nextStep, done: nextStep >= STEPS.length }
    writeTourState(next)
    setPos(null)
    setTour(next)
  }, [tour.step])

  const skip = useCallback(() => {
    const next: TourState = { step: tour.step, done: true }
    writeTourState(next)
    setTour(next)
  }, [tour.step])

  useEffect(() => {
    if (!active) return
    const el = document.querySelector(`[data-tour="${active.anchor}"]`)
    if (!(el instanceof HTMLElement)) {
      // Anchor not on this screen — skip the step rather than block the tour.
      advance()
      return
    }
    const update = () => setPos(computePosition(el.getBoundingClientRect()))
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [active, advance])

  if (!active || !pos) return null

  const isLastOnPage = STEPS[tour.step + 1]?.page !== page

  return (
    <div
      role="dialog"
      aria-label={active.title}
      className="fixed z-[60] bg-surface border border-line rounded-lg elev-3 p-4 animate-fade-up"
      style={{ top: pos.top, left: pos.left, width: CARD_WIDTH }}
    >
      <span
        aria-hidden
        className={`absolute w-3 h-3 bg-surface border-line rotate-45 ${
          pos.placement === 'below' ? '-top-[7px] border-l border-t' : '-bottom-[7px] border-r border-b'
        }`}
        style={{ left: pos.caretLeft }}
      />
      <p className="text-[10px] font-medium text-ink-faint uppercase tracking-[0.14em] tnum">
        Step {tour.step + 1} of {STEPS.length}
      </p>
      <h3 className="text-sm font-semibold text-ink mt-1">{active.title}</h3>
      <p className="text-sm text-ink-soft mt-1 leading-relaxed">{active.body}</p>
      <div className="flex items-center gap-2 mt-3">
        <Button size="sm" onClick={advance}>
          {isLastOnPage ? 'Done' : 'Next'}
        </Button>
        <Button size="sm" variant="ghost" onClick={skip}>
          Skip tour
        </Button>
      </div>
    </div>
  )
}
