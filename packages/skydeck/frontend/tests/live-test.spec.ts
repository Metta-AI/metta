import { test, expect } from '@playwright/test';

// This test runs against the actual backend
test.describe('Live Application Test', () => {
  test('check select all in real groups', async ({ page }) => {
    // Listen to console messages for debugging
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(msg.text());
      console.log('BROWSER CONSOLE:', msg.text());
    });

    // Go to the actual running application
    await page.goto('http://localhost:8000/');

    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    // Take a screenshot of the initial state
    await page.screenshot({ path: 'test-results/live-initial.png', fullPage: true });

    // Find all group headers and log them
    const groups = await page.locator('.experiment-group').all();
    console.log(`Found ${groups.length} experiment groups`);

    // Try to find and click the first group's select-all checkbox
    if (groups.length > 0) {
      const firstGroup = groups[0];

      // Get group name
      const groupName = await firstGroup.locator('.group-header').textContent();
      console.log('First group name:', groupName);

      // Find the select-all checkbox
      const selectAllCheckbox = firstGroup.locator('th.col-checkbox input[type="checkbox"]').first();

      // Check if it exists
      const exists = await selectAllCheckbox.count();
      console.log('Select-all checkbox count:', exists);

      if (exists > 0) {
        // Take screenshot before clicking
        await page.screenshot({ path: 'test-results/live-before-select-all.png', fullPage: true });

        // Click it
        await selectAllCheckbox.click();
        await page.waitForTimeout(500);

        // Take screenshot after clicking
        await page.screenshot({ path: 'test-results/live-after-select-all.png', fullPage: true });

        // Check if any experiments got selected
        const selectedCheckboxes = await firstGroup.locator('tr.main-row input[type="checkbox"]:checked').count();
        console.log('Selected checkboxes after clicking select-all:', selectedCheckboxes);

        // Log console messages
        console.log('\nAll console messages:', consoleMessages);
      }
    }
  });
});
