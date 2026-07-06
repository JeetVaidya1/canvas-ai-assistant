import { AnimatePresence, motion } from 'motion/react'
import { BookOpen, Plus, Trash2, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import ErrorInline from '@/components/shared/ErrorInline'
import type { ChatSession } from '@/lib/api'

interface HistoryDrawerProps {
  open: boolean
  onClose: () => void
  sessions: ReadonlyArray<ChatSession>
  loadFailed: boolean
  onRetryLoad: () => void
  activeSessionId?: string
  onSelect: (session: ChatSession) => void
  onDelete: (session: ChatSession) => void
  onNewChat: () => void
  courseTitle: string
  fileCount: number
}

/** Chat history slide-over — sessions list with delete, plus course context footer. */
export function HistoryDrawer({
  open,
  onClose,
  sessions,
  loadFailed,
  onRetryLoad,
  activeSessionId,
  onSelect,
  onDelete,
  onNewChat,
  courseTitle,
  fileCount,
}: HistoryDrawerProps) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 z-30 bg-ink/30"
          />
          <motion.aside
            initial={{ x: -320 }}
            animate={{ x: 0 }}
            exit={{ x: -320 }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            className="absolute inset-y-0 left-0 z-40 flex w-[320px] flex-col border-r border-line bg-surface elev-3"
          >
            <div className="flex h-14 items-center justify-between border-b border-line px-4">
              <span className="text-sm font-semibold text-ink">Chat history</span>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-ink-faint transition-colors hover:bg-paper-deep hover:text-ink"
                aria-label="Close history"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="px-3 py-3">
              <Button variant="secondary" onClick={onNewChat} leftIcon={<Plus className="h-4 w-4" />} className="w-full">
                New chat
              </Button>
            </div>
            <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
              {loadFailed ? (
                <ErrorInline
                  message="Couldn't load your chat history."
                  onRetry={onRetryLoad}
                  className="mx-1 mt-2"
                />
              ) : sessions.length === 0 ? (
                <p className="px-2 py-6 text-center text-xs text-ink-faint">No conversations yet.</p>
              ) : (
                sessions.map((session) => {
                  const active = activeSessionId === session.id
                  return (
                    <div
                      key={session.id}
                      onClick={() => onSelect(session)}
                      className={cn(
                        'group flex cursor-pointer items-center justify-between gap-2 rounded-lg border px-2.5 py-2 transition-colors',
                        active
                          ? 'border-accent-line bg-accent-wash text-ink'
                          : 'border-transparent text-ink-soft hover:bg-paper-deep hover:text-ink',
                      )}
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">{session.title || 'Untitled chat'}</p>
                        <p className="text-[11px] text-ink-faint">
                          {new Date(session.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onDelete(session)
                        }}
                        className="p-1 text-ink-faint opacity-0 transition-all hover:text-danger group-hover:opacity-100"
                        aria-label="Delete chat"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )
                })
              )}
            </div>
            <div className="border-t border-line p-3">
              <div className="flex items-center gap-2 text-xs text-ink-faint">
                <BookOpen className="h-3.5 w-3.5" />
                {courseTitle} · {fileCount > 0 ? `${fileCount} files` : 'no files'}
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
