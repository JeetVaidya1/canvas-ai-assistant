// src/lib/api/notes.ts
import { BASE_URL, apiFetch } from './client'

/** ===== Notes ===== */

export interface Flashcard { q: string; a: string }

export interface NotesResponse {
  status: 'success' | 'error'
  notes: string
  suggested_title: string
  word_count: number
  reading_time: string
  topics: string[]
  source_files: string[]
  message?: string
  flashcards?: { q: string; a: string }[]
}

export interface SavedNote {
  id: string
  course_id: string
  title: string
  content: string
  source_files: string[]
  topic_focus: string
  topics: string[]
  word_count: number
  reading_time: string
  created_at: string
  updated_at: string
}

/**
 * Long-running endpoint — uses apiFetch with extended timeout.
 */
export async function generateNotes(
  courseId: string,
  fileNames: string[],
  topic: string = '',
  style: 'detailed' | 'summary' | 'outline' = 'detailed'
): Promise<NotesResponse> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('file_names', JSON.stringify(fileNames));
  form.append('topic', topic);
  form.append('style', style);

  const data = await apiFetch('/generate-notes', { method: 'POST', body: form }, 300_000);
  if (data.status !== 'success') {
    throw new Error(data.message || 'Notes generation failed');
  }
  return data;
}

export async function saveNotes(
  courseId: string,
  title: string,
  content: string,
  sourceFiles: string[],
  topic: string = '',
  noteId?: string
): Promise<SavedNote> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('title', title);
  form.append('content', content);
  form.append('source_files', JSON.stringify(sourceFiles));
  form.append('topic', topic);
  if (noteId) form.append('note_id', noteId);

  const data = await apiFetch('/save-notes', { method: 'POST', body: form });
  return data.note;
}

export async function updateNote(
  noteId: string,
  courseId: string,
  title: string,
  content: string,
  sourceFiles: string[] = [],
  topic: string = ''
): Promise<SavedNote> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('title', title);
  form.append('content', content);
  form.append('source_files', JSON.stringify(sourceFiles));
  form.append('topic', topic);

  const data = await apiFetch(`/notes/${encodeURIComponent(noteId)}`, { method: 'PUT', body: form });
  return data.note;
}

export async function getNotes(courseId: string): Promise<SavedNote[]> {
  const data = await apiFetch(`/notes/${encodeURIComponent(courseId)}`);
  return data.notes || [];
}

export async function deleteNotes(noteId: string): Promise<void> {
  await apiFetch(`/notes/${encodeURIComponent(noteId)}`, { method: 'DELETE' });
}

/** ===== Export ===== */
export async function exportNotesPdf(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-notes-pdf/${encodeURIComponent(courseId)}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}
