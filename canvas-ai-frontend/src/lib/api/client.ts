// src/lib/api/client.ts
import { getAccessToken } from '../auth'

/** ===== Backend base + helper ===== */
export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function apiFetch(path: string, init?: RequestInit, timeoutMs = 60_000) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    // Attach the Supabase access token so the backend can verify identity.
    const token = await getAccessToken()
    const headers = new Headers(init?.headers)
    if (token) headers.set('Authorization', `Bearer ${token}`)
    const resp = await fetch(`${BASE_URL}${path}`, { ...init, headers, signal: ctrl.signal })
    // Most FastAPI errors include .detail; fall back to statusText
    if (!resp.ok) {
      let msg = resp.statusText
      try {
        const body = await resp.json()
        msg = body?.detail || body?.message || msg
      } catch { /* ignore json parse errors */ }
      throw new Error(msg || 'Request failed')
    }
    // Some endpoints return no body (void)
    const text = await resp.text()
    return text ? JSON.parse(text) : {}
  } finally {
    clearTimeout(timer)
  }
}
