import { test, expect } from '@playwright/test';

// This test duplicates a specific experiment in the live app
test.describe('Live Duplicate Test', () => {
  test('duplicate daveey.ca5.4x4.lr_16.ppo experiment', async ({ page }) => {
    // Listen to console messages for debugging
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(msg.text());
      console.log('BROWSER CONSOLE:', msg.text());
    });

    // Listen for network errors
    page.on('pageerror', error => {
      console.log('PAGE ERROR:', error);
    });

    // Listen to all network requests
    page.on('response', async response => {
      const url = response.url();
      const status = response.status();
      if (url.includes('/api/')) {
        console.log(`API ${response.request().method()} ${url} - ${status}`);
        if (status >= 400) {
          const body = await response.text().catch(() => '(could not read body)');
          console.log('ERROR RESPONSE BODY:', body);
        }
      }
    });

    // Go to the actual running application
    await page.goto('http://localhost:8000/');

    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // Take a screenshot of the initial state
    await page.screenshot({ path: 'test-results/duplicate-live-initial.png', fullPage: true });

    // Find the experiment row for "daveey.ca5.4x4.lr_16.ppo"
    const expRow = page.locator('tr.main-row').filter({ hasText: 'daveey.ca5.4x4.lr_16.ppo' });

    // Check if it exists
    const exists = await expRow.count();
    console.log('Found experiment rows:', exists);

    if (exists === 0) {
      console.log('Experiment not found!');
      return;
    }

    // Find the checkbox for this experiment
    const checkbox = expRow.locator('input[type="checkbox"]').first();

    // Check the checkbox
    await checkbox.check();
    await page.waitForTimeout(500);

    // Take screenshot after selecting
    await page.screenshot({ path: 'test-results/duplicate-live-selected.png', fullPage: true });

    // Find and click the duplicate button
    const duplicateButton = page.locator('.bulk-actions button:has-text("Duplicate")');
    const buttonExists = await duplicateButton.count();
    console.log('Duplicate button exists:', buttonExists > 0);

    if (buttonExists > 0) {
      await duplicateButton.click();

      // Wait for the operation to complete
      await page.waitForTimeout(3000);

      // Take screenshot after duplicate
      await page.screenshot({ path: 'test-results/duplicate-live-after.png', fullPage: true });

      // Check for notifications
      const successNotification = page.locator('.notification-success');
      const errorNotification = page.locator('.notification-error');

      const hasSuccess = await successNotification.count();
      const hasError = await errorNotification.count();

      console.log('Success notification:', hasSuccess > 0);
      console.log('Error notification:', hasError > 0);

      if (hasError > 0) {
        const errorText = await errorNotification.textContent();
        console.log('ERROR TEXT:', errorText);
      }

      if (hasSuccess > 0) {
        const successText = await successNotification.textContent();
        console.log('SUCCESS TEXT:', successText);
      }

      // Log all console messages
      console.log('\nAll console messages:', consoleMessages);
    }
  });
});
