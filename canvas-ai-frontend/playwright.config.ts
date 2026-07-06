import { defineConfig } from '@playwright/test'

/**
 * Smoke suite: unauthenticated flows only (landing, login, auth redirects).
 * Uses the installed Chrome so CI/dev don't need a browser download.
 * Authenticated journeys are exercised manually + via vitest for now.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:5173',
    channel: 'chrome',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
})
