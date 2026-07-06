import ReviewPanel from '@/components/ReviewPanel'

interface ReviewQueueProps {
  courseId: string
  userId: string
}

/**
 * Mistake-driven review queue section. ReviewPanel manages its own
 * visibility: it stays out of the way while loading or when nothing is due,
 * and runs grading against a local snapshot so background refetches
 * (grading invalidates reviews + readiness) never reshuffle a live session.
 */
export function ReviewQueue({ courseId, userId }: ReviewQueueProps) {
  return <ReviewPanel courseId={courseId} userId={userId} />
}
