import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Badge } from '@/components/ui/Badge'
import { Input, Textarea } from '@/components/ui/Input'
import { ProgressBar } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { EmptyState, ErrorState } from '@/components/ui/States'

describe('Badge', () => {
  it('renders children with the requested tone classes', () => {
    render(<Badge tone="danger">3 overdue</Badge>)
    const el = screen.getByText('3 overdue')
    expect(el).toBeInTheDocument()
    expect(el.className).toContain('text-danger')
  })
})

describe('Input', () => {
  it('associates label with the field and shows hint', () => {
    render(<Input label="Email" hint="School email preferred" />)
    expect(screen.getByLabelText('Email')).toBeInTheDocument()
    expect(screen.getByText('School email preferred')).toBeInTheDocument()
  })

  it('shows error instead of hint and sets aria-invalid', () => {
    render(<Input label="Email" hint="hidden when erroring" error="Required" />)
    expect(screen.getByRole('alert')).toHaveTextContent('Required')
    expect(screen.queryByText('hidden when erroring')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Email')).toHaveAttribute('aria-invalid', 'true')
  })

  it('forwards typing to onChange', () => {
    const onChange = vi.fn()
    render(<Input label="Name" onChange={onChange} />)
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Jeet' } })
    expect(onChange).toHaveBeenCalledOnce()
  })
})

describe('Textarea', () => {
  it('renders with label', () => {
    render(<Textarea label="Notes" />)
    expect(screen.getByLabelText('Notes')).toBeInTheDocument()
  })
})

describe('scoreTone', () => {
  it('maps scores to semantic tones', () => {
    expect(scoreTone(85).label).toBe('On track')
    expect(scoreTone(55).label).toBe('Getting there')
    expect(scoreTone(20).label).toBe('Needs work')
  })

  it('boundaries: 70 is on track, 40 is getting there', () => {
    expect(scoreTone(70).label).toBe('On track')
    expect(scoreTone(40).label).toBe('Getting there')
    expect(scoreTone(39.9).label).toBe('Needs work')
  })
})

describe('ProgressBar', () => {
  it('exposes clamped progressbar semantics', () => {
    render(<ProgressBar value={140} label="Mastery" />)
    const bar = screen.getByRole('progressbar', { name: 'Mastery' })
    expect(bar).toHaveAttribute('aria-valuenow', '100')
  })
})

describe('EmptyState', () => {
  it('renders title, description and action', () => {
    render(<EmptyState title="No courses yet" description="Create one to begin." action={<button>New course</button>} />)
    expect(screen.getByText('No courses yet')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New course' })).toBeInTheDocument()
  })
})

describe('ErrorState', () => {
  it('calls onRetry', () => {
    const onRetry = vi.fn()
    render(<ErrorState onRetry={onRetry} />)
    fireEvent.click(screen.getByRole('button', { name: /try again/i }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('compact variant renders inline alert', () => {
    render(<ErrorState compact title="Failed to load topics" onRetry={() => {}} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Failed to load topics')
  })
})
