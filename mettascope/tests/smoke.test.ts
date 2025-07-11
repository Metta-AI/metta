import { test, expect } from '@playwright/test'

const host = 'http://localhost:8000'

test('smoke test', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text())
    }
  })
  await page.goto(host)
  expect(consoleErrors).toHaveLength(0)
})

test('load a replay', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text())
    }
  })
  await page.goto(`${host}/?wsUrl=%2Fws`)

  // Wait for the page to fully load the replay and render the first frame
  await page.waitForFunction(
    () => {
      const state = (window as any).state
      return state && state.replay !== null
    },
    { timeout: 10000 }
  )
  expect(consoleErrors).toHaveLength(0)
})

test('load a replay and play it', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text())
    }
  })
  await page.goto(`${host}/?wsUrl=%2Fws&play=true`)

  // Wait for the page to fully load the replay and render the first frame
  await page.waitForFunction(
    () => {
      const state = (window as any).state
      return state && state.isPlaying == true
    },
    { timeout: 10000 }
  )
  expect(consoleErrors).toHaveLength(0)
})
