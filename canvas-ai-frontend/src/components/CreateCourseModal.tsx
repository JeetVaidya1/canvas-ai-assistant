import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateCourse } from '@/hooks/useCourses'
import { Modal } from '@/components/ui/Modal'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { showError } from '@/lib/toast'

/** "Intro to Computer Science" → "intro-to-computer-science" (max 32 chars). */
function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32)
}

interface CreateCourseModalProps {
  open: boolean
  onClose: () => void
}

/**
 * Create-course dialog. The course ID is derived from the title automatically
 * (still editable for people who care), so the common path is one field + Enter.
 */
export default function CreateCourseModal({ open, onClose }: CreateCourseModalProps) {
  const navigate = useNavigate()
  const createCourse = useCreateCourse()
  const [title, setTitle] = useState('')
  const [courseId, setCourseId] = useState('')
  const [idTouched, setIdTouched] = useState(false)

  const effectiveId = idTouched ? courseId : slugify(title)
  const canSubmit = title.trim().length > 0 && effectiveId.length > 0 && !createCourse.isPending

  const reset = () => {
    setTitle('')
    setCourseId('')
    setIdTouched(false)
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const handleCreate = async () => {
    if (!canSubmit) return
    try {
      await createCourse.mutateAsync({ courseId: effectiveId, title: title.trim() })
      reset()
      onClose()
      navigate(`/course/${effectiveId}`)
    } catch {
      showError('Couldn’t create the course — that ID may already be taken.')
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Create a course"
      description="Name it after the class you’re taking — you’ll add materials next."
      footer={
        <>
          <Button variant="ghost" onClick={handleClose}>Cancel</Button>
          <Button onClick={handleCreate} loading={createCourse.isPending} disabled={!canSubmit}>
            Create course
          </Button>
        </>
      }
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault()
          void handleCreate()
        }}
      >
        <Input
          data-autofocus
          label="Course title"
          placeholder="e.g. Introduction to Computer Science"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <Input
          label="Course ID"
          hint="Used in links — generated from the title, edit if you like."
          placeholder="intro-to-computer-science"
          value={effectiveId}
          onChange={(e) => {
            setIdTouched(true)
            setCourseId(slugify(e.target.value))
          }}
        />
        {/* Hidden submit so Enter works from either field */}
        <button type="submit" className="hidden" aria-hidden="true" tabIndex={-1} />
      </form>
    </Modal>
  )
}
