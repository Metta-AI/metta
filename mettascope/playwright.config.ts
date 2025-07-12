import { defineConfig, devices } from '@playwright/test'

const gpuFlags = [
  '--no-sandbox',
  '--headless=new',
  '--use-angle=vulkan',
  '--enable-features=Vulkan',
  '--disable-vulkan-surface',
  '--enable-unsafe-webgpu',
]

export default defineConfig({
  testDir: './tests',
  outputDir: './tests/test-results',
  use: {
    screenshot: 'on',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chromium',
        launchOptions: { args: gpuFlags },
      },
    },
  ],
})
