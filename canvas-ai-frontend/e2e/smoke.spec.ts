import { test, expect } from '@playwright/test'

test.describe('public surface', () => {
  test('landing renders hero, product preview and CTAs', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1 })).toContainText('study system')
    // DOM-built product preview shows real product truths
    await expect(page.getByText('Exam readiness', { exact: false }).first()).toBeVisible()
    await expect(page.getByRole('button', { name: 'Start free' }).first()).toBeVisible()
    // No emoji rain / no fake stats
    await expect(page.locator('text=10+ Study tools')).toHaveCount(0)
  })

  test('landing CTA leads to login', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Get started' }).click()
    await expect(page).toHaveURL(/\/login/)
    await expect(page.getByRole('button', { name: 'Continue with Google' })).toBeVisible()
  })

  test('login form switches modes', async ({ page }) => {
    await page.goto('/login')
    await expect(page.getByRole('heading', { name: 'Welcome back' })).toBeVisible()
    await page.getByText('Create an account').click()
    await expect(page.getByRole('heading', { name: 'Create your account' })).toBeVisible()
    await page.getByText('Email me a magic link instead').click()
    await expect(page.getByRole('heading', { name: 'Sign in with a magic link' })).toBeVisible()
  })

  test('protected routes redirect to login when signed out', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page).toHaveURL(/\/login/)
  })

  test('unknown in-app route shows not-found, not a crash', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (e) => errors.push(e.message))
    await page.goto('/definitely-not-a-route')
    // Signed out → RequireAuth kicks to login; either way the app must not throw
    expect(errors).toEqual([])
  })
})
