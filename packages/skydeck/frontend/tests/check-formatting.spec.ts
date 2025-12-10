import { test, expect } from '@playwright/test';

test('check large number formatting', async ({ page }) => {
  await page.goto('http://localhost:8000/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);

  // Find the experiment with large number
  const expRow = page.locator('tr.main-row').filter({ hasText: 'lr_32' });

  // Click to expand
  await expRow.click();
  await page.waitForTimeout(1000);

  // Take a screenshot
  await page.screenshot({ path: 'test-results/formatting-check.png', fullPage: true });

  // Check the displayed value in the configuration panel
  const timestepsValue = await page.locator('text=trainer.total_timesteps').locator('..').locator('td').last().textContent();
  console.log('Displayed timesteps value:', timestepsValue);

  // Check in the main table (flag column)
  const tableCell = await expRow.locator('.col-flag .flag-value').filter({ hasText: /B$|M$|K$|[0-9]/ }).first().textContent();
  console.log('Table cell value:', tableCell);
});
