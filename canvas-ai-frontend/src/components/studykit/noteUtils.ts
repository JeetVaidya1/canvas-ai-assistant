// src/components/studykit/noteUtils.ts — shared types + pure helpers for the Study Kit
import type { SavedNote } from '@/lib/api'

export type Flashcard = { q: string; a: string }

export type NoteStyle = 'detailed' | 'summary' | 'outline'

/** Segmented control options for note generation style. */
export const NOTE_STYLES: { value: NoteStyle; label: string; hint: string }[] = [
  { value: 'detailed', label: 'Detailed', hint: 'Comprehensive, in-depth notes' },
  { value: 'summary', label: 'Summary', hint: 'Key points only' },
  { value: 'outline', label: 'Outline', hint: 'Structured outline format' },
]

/**
 * Honest staged narrative for the in-flight generation state. These describe
 * what the backend is actually doing — never rendered as fake percentages.
 */
export const LOADING_STAGES = [
  'Retrieving from your materials…',
  'Grounding key concepts…',
  'Writing your notes…',
  'Generating flashcards…',
]

/** Forgiving parser kept ONLY as a fallback for legacy saved notes whose
 *  flashcards were embedded inline. New generations use response.flashcards. */
export const parseFlashcardsFromText = (text: string): Flashcard[] => {
  try {
    if (!text) return []
    const headerMatch = /flashcards(?:\s*\(json\))?/i.exec(text)
    if (headerMatch) {
      let i = headerMatch.index + headerMatch[0].length
      while (i < text.length && /\s|:/.test(text[i])) i++
      if (text.slice(i, i + 3) === '```') {
        i += 3
        if (/^json/i.test(text.slice(i, i + 4))) i += 4
        while (i < text.length && /\s/.test(text[i])) i++
        const fence = text.indexOf('```', i)
        const raw = fence !== -1 ? text.slice(i, fence) : text.slice(i)
        try {
          const arr = JSON.parse(raw.replace(/,\s*([\]}])/g, '$1'))
          if (Array.isArray(arr)) {
            return arr
              .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
              .filter((it) => it.q && it.a)
          }
        } catch {/* fall through */}
      }
      const start = text.indexOf('[', i)
      if (start !== -1) {
        let depth = 0
        let end = -1
        for (let j = start; j < text.length; j++) {
          const ch = text[j]
          if (ch === '[') depth++
          else if (ch === ']') {
            depth--
            if (depth === 0) { end = j; break }
          }
        }
        if (end !== -1) {
          try {
            const arr = JSON.parse(text.slice(start, end + 1).replace(/,\s*([\]}])/g, '$1'))
            if (Array.isArray(arr)) {
              return arr
                .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
                .filter((it) => it.q && it.a)
            }
          } catch {/* fall through */}
        }
      }
    }
    const anyArr = text.match(/\[[\s\S]+?\]/)
    if (anyArr) {
      try {
        const arr = JSON.parse(anyArr[0].replace(/,\s*([\]}])/g, '$1'))
        if (Array.isArray(arr)) {
          return arr
            .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
            .filter((it) => it.q && it.a)
        }
      } catch {/* ignore */}
    }
    return []
  } catch {
    return []
  }
}

export const getWordCount = (text: string): number =>
  text.trim() ? text.trim().split(/\s+/).length : 0

export const getReadingTime = (wordCount: number): string =>
  `${Math.max(1, Math.ceil(wordCount / 200))} min read`

/** Download a saved note as a plain-text file. */
export const downloadNoteAsText = (note: SavedNote): void => {
  const element = document.createElement('a')
  const file = new Blob([note.content || ''], { type: 'text/plain' })
  element.href = URL.createObjectURL(file)
  element.download = `${(note.title || 'notes').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`
  document.body.appendChild(element)
  element.click()
  document.body.removeChild(element)
  URL.revokeObjectURL(element.href)
}
