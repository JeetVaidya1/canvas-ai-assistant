import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'

type ModalSize = 'sm' | 'md' | 'lg'

const sizeClass: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
}

interface ModalProps {
  open: boolean
  onClose: () => void
  title?: string
  description?: string
  size?: ModalSize
  /** Prevent closing via backdrop / Esc (e.g. during a destructive submit). */
  locked?: boolean
  children: ReactNode
  footer?: ReactNode
}

/**
 * Accessible modal dialog: portal, backdrop, Esc to close, focus containment,
 * body scroll lock. Every dialog in the app goes through this — no bespoke
 * overlays.
 */
export function Modal({ open, onClose, title, description, size = 'md', locked = false, children, footer }: ModalProps) {
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const previouslyFocused = document.activeElement as HTMLElement | null
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !locked) onClose()
      if (e.key === 'Tab') {
        const focusables = panelRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        )
        if (!focusables || focusables.length === 0) return
        const first = focusables[0]
        const last = focusables[focusables.length - 1]
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    // Move focus into the dialog
    requestAnimationFrame(() => panelRef.current?.querySelector<HTMLElement>('[data-autofocus]')?.focus())
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
      previouslyFocused?.focus()
    }
  }, [open, locked, onClose])

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label={title}>
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}
            onClick={locked ? undefined : onClose}
          />
          <motion.div
            ref={panelRef}
            className={cn('relative w-full rounded-xl bg-bg-overlay border border-border elev-3 flex flex-col max-h-[85vh]', sizeClass[size])}
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 8 }}
            transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
          >
            {(title || !locked) && (
              <div className="flex items-start justify-between gap-4 px-5 pt-5 pb-1 flex-shrink-0">
                <div className="min-w-0">
                  {title && <h2 className="text-base font-semibold text-zinc-50">{title}</h2>}
                  {description && <p className="text-sm text-zinc-400 mt-1">{description}</p>}
                </div>
                {!locked && (
                  <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close" className="-mr-2 -mt-1 px-2">
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
            )}
            <div className="px-5 py-4 overflow-y-auto">{children}</div>
            {footer && (
              <div className="flex items-center justify-end gap-2.5 px-5 py-4 border-t border-border-subtle flex-shrink-0">
                {footer}
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  )
}
