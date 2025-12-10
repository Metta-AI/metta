import { test } from '@playwright/test';

test('simple check', async ({ page }) => {
  await page.goto('http://localhost:8000/');
  await page.waitForTimeout(5000);

  // Take a screenshot
  await page.screenshot({ path: 'test-results/simple-check.png', fullPage: true });
});
