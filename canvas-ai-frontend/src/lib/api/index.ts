// src/lib/api/index.ts
// Barrel: re-exports every domain module so `@/lib/api` resolves identically
// to the original flat api.ts. Importers do not need to change.
export * from './courses'
export * from './chat'
export * from './quiz'
export * from './sharing'
export * from './tutor'
export * from './integrations'
export * from './reviews'
export * from './analytics'
export * from './flashcards'
export * from './notes'
export * from './practice'
export * from './exams'
export * from './planner'
export * from './audio'
