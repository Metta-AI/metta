import { test, expect } from '@playwright/test';

test('check status of duplicate', async ({ page }) => {
  await page.goto('http://localhost:8000/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // Take a screenshot
  await page.screenshot({ path: 'test-results/current-state.png', fullPage: true });

  // Find the duplicate experiment
  const dupRow = page.locator('tr.main-row').filter({ hasText: '(copy)' });
  const exists = await dupRow.count();

  if (exists > 0) {
    const statusText = await dupRow.locator('.col-state').textContent();
    console.log('Duplicate status:', statusText);
  }
});
