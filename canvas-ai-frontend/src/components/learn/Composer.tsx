import { useEffect } from 'react'
import type { FormEvent, KeyboardEvent, RefObject } from 'react'
import { ArrowUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'

interface ComposerProps {
  /** 'hero' = center-first empty state; 'docked' = pinned under transcript. */
  variant: 'hero' | 'docked'
  value: string
  onChange: (value: string) => void
  onSend: () => void
  sending: boolean
  hasFiles: boolean
  textareaRef: RefObject<HTMLTextAreaElement | null>
}

/**
 * The Learn chat composer — identical chrome in both positions so the
 * dock-on-send transition reads as the same element moving.
 */
export function Composer({ variant, value, onChange, onSend, sending, hasFiles, textareaRef }: ComposerProps) {
  // Auto-grow the composer textarea.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value, textareaRef])

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSend()
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        'relative flex w-full items-end gap-2 rounded-[20px] border border-white/12 bg-white/[0.03] p-2.5 shadow-lg transition-all',
        'focus-within:border-cyan-400/60 focus-within:bg-white/[0.05] focus-within:glow-brand-sm',
      )}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={variant === 'hero' ? 2 : 1}
        autoFocus={variant === 'hero'}
        placeholder={hasFiles ? 'Ask anything about your course…' : 'Ask a question…'}
        className="max-h-[200px] flex-1 resize-none bg-transparent px-2.5 py-2 text-[15px] text-zinc-100 placeholder-zinc-500 outline-none"
      />
      <Button
        type="submit"
        variant="primary"
        loading={sending}
        disabled={sending || !value.trim()}
        className="!h-10 !w-10 flex-shrink-0 !rounded-xl !p-0"
        aria-label="Send message"
      >
        {!sending && <ArrowUp className="h-[18px] w-[18px]" />}
      </Button>
    </form>
  )
}
