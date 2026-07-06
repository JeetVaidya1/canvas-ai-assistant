import { forwardRef, useId } from 'react'
import type { InputHTMLAttributes, TextareaHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

const fieldBase =
  'w-full bg-surface border rounded-lg text-ink placeholder-ink-faint text-sm ' +
  'outline-none transition-colors focus:ring-2 ' +
  'disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-paper-deep'

const fieldTone = {
  normal: 'border-line hover:border-line-strong focus:border-accent focus:ring-accent/20',
  error: 'border-danger/50 focus:border-danger focus:ring-danger/20',
}

interface FieldChromeProps {
  label?: string
  hint?: string
  error?: string
  id: string
  children: ReactNode
}

/** Shared label/hint/error chrome so every form field reads identically. */
function FieldChrome({ label, hint, error, id, children }: FieldChromeProps) {
  return (
    <div className="space-y-1.5">
      {label && (
        <label htmlFor={id} className="block text-xs font-medium text-ink-soft">
          {label}
        </label>
      )}
      {children}
      {error ? (
        <p className="text-xs text-danger" role="alert">{error}</p>
      ) : hint ? (
        <p className="text-xs text-ink-faint">{hint}</p>
      ) : null}
    </div>
  )
}

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
  error?: string
  leftIcon?: ReactNode
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { label, hint, error, leftIcon, className, id: idProp, ...rest },
  ref,
) {
  const autoId = useId()
  const id = idProp ?? autoId
  return (
    <FieldChrome label={label} hint={hint} error={error} id={id}>
      <div className="relative">
        {leftIcon && (
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none [&>svg]:w-4 [&>svg]:h-4">
            {leftIcon}
          </span>
        )}
        <input
          ref={ref}
          id={id}
          aria-invalid={error ? true : undefined}
          className={cn(fieldBase, error ? fieldTone.error : fieldTone.normal, 'px-3 py-2', leftIcon && 'pl-9', className)}
          {...rest}
        />
      </div>
    </FieldChrome>
  )
})

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  hint?: string
  error?: string
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { label, hint, error, className, id: idProp, rows = 4, ...rest },
  ref,
) {
  const autoId = useId()
  const id = idProp ?? autoId
  return (
    <FieldChrome label={label} hint={hint} error={error} id={id}>
      <textarea
        ref={ref}
        id={id}
        rows={rows}
        aria-invalid={error ? true : undefined}
        className={cn(fieldBase, error ? fieldTone.error : fieldTone.normal, 'px-3 py-2 resize-y min-h-[2.5rem]', className)}
        {...rest}
      />
    </FieldChrome>
  )
})
