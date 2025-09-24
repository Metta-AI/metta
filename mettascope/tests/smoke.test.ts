import { type Page, test } from '@playwright/test'

const host = 'http://localhost:8000'

type ErrorCollections = {
  consoleErrors: string[]
  networkErrors: string[]
}

function trackPageErrors(page: Page): ErrorCollections {
  const consoleErrors: string[] = []
  const networkErrors: string[] = []

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text())
    }
  })

  page.on('requestfailed', (request) => {
    const failure = request.failure()
    networkErrors.push(`FAILED ${request.method()} ${request.url()}${failure ? ` – ${failure.errorText}` : ''}`)
  })

  page.on('response', (response) => {
    const status = response.status()
    if (status >= 400) {
      networkErrors.push(`STATUS ${status} ${response.url()}`)
    }
  })

  return { consoleErrors, networkErrors }
}

function expectNoErrors(errors: ErrorCollections) {
  if (errors.networkErrors.length > 0) {
    throw new Error(`Network errors detected:\n${errors.networkErrors.join('\n')}`)
  }
  if (errors.consoleErrors.length > 0) {
    throw new Error(`Console errors detected:\n${errors.consoleErrors.join('\n')}`)
  }
}

test('smoke test', async ({ page }) => {
  const errors = trackPageErrors(page)
  await page.goto(host)
  expectNoErrors(errors)
})

test('load a replay', async ({ page }) => {
  const errors = trackPageErrors(page)
  await page.goto(`${host}/?wsUrl=%2Fws`)

  // Wait for the page to fully load the replay and render the first frame
  await page.waitForFunction(
    () => {
      const state = (window as any).state
      return state && state.replay !== null
    },
    { timeout: 10000 }
  )
  expectNoErrors(errors)
})

test('load a replay and play it', async ({ page }) => {
  const errors = trackPageErrors(page)
  await page.goto(`${host}/?wsUrl=%2Fws&play=true`)

  // Wait for the page to fully load the replay and render the first frame
  await page.waitForFunction(
    () => {
      const state = (window as any).state
      return state && state.isPlaying === true
    },
    { timeout: 10000 }
  )
  expectNoErrors(errors)
})
