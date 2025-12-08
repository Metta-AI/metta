import { test, expect } from '@playwright/test';

test.describe('Duplicate and Select Functionality', () => {
  test.beforeEach(async ({ page }) => {
    // Mock health endpoint
    await page.route('/api/health', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          num_experiments: 3,
          num_running_jobs: 1,
          skypilot: { staleness_seconds: 10.5 },
          s3: { staleness_seconds: 5.2 },
          observatory: { staleness_seconds: 15.3 },
        }),
      });
    });

    // Mock groups endpoint with a group containing experiments
    await page.route('/api/groups', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          groups: [
            {
              id: 'test-group-1',
              name: 'Test Group',
              flags: ['trainer.batch_size', 'model.hidden_dim'],
              order: 0,
              collapsed: false,
              experiments: [
                {
                  id: '1',
                  name: 'experiment-1',
                  base_command: 'lt',
                  tool_path: 'recipes.experiment.test',
                  git_branch: 'main',
                  nodes: 2,
                  gpus: 4,
                  instance_type: null,
                  cloud: null,
                  spot: false,
                  flags: { 'trainer.batch_size': 32, 'model.hidden_dim': 256 },
                  description: 'Test experiment 1',
                  tags: [],
                  group: 'test-group-1',
                  desired_state: 'RUNNING',
                  current_state: 'RUNNING',
                  current_job_id: 'job-1',
                  starred: false,
                  is_expanded: false,
                  latest_epoch: 100,
                  exp_order: 0,
                },
                {
                  id: '2',
                  name: 'experiment-2',
                  base_command: 'lt',
                  tool_path: 'recipes.experiment.test',
                  git_branch: 'main',
                  nodes: 1,
                  gpus: 2,
                  instance_type: null,
                  cloud: null,
                  spot: false,
                  flags: { 'trainer.batch_size': 64, 'model.hidden_dim': 128 },
                  description: 'Test experiment 2',
                  tags: [],
                  group: 'test-group-1',
                  desired_state: 'STOPPED',
                  current_state: 'STOPPED',
                  current_job_id: null,
                  starred: false,
                  is_expanded: false,
                  latest_epoch: 50,
                  exp_order: 1,
                },
              ],
            },
          ],
          ungrouped: [
            {
              id: '3',
              name: 'ungrouped-experiment',
              base_command: 'lt',
              tool_path: 'recipes.experiment.test',
              git_branch: 'feature',
              nodes: 1,
              gpus: 1,
              instance_type: null,
              cloud: null,
              spot: true,
              flags: { 'trainer.batch_size': 16 },
              description: null,
              tags: [],
              group: null,
              desired_state: 'STOPPED',
              current_state: 'STOPPED',
              current_job_id: null,
              starred: false,
              is_expanded: false,
              latest_epoch: 10,
              exp_order: 0,
            },
          ],
        }),
      });
    });

    await page.route('/api/skypilot-jobs*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ jobs: [] }),
      });
    });

    await page.route('/api/settings/jobs-filters', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ value: { showStopped: false, showOrphaned: false, limit: 20 } }),
      });
    });

    await page.goto('/');
  });

  test('select all checkbox in group selects all experiments in that group', async ({ page }) => {
    // Wait for the group to load
    await expect(page.locator('text=Test Group')).toBeVisible();

    // Find the select-all checkbox in the group
    // Strategy: Find the experiment-group div that contains "Test Group" text,
    // then find the checkbox within its table header
    const groupDiv = page.locator('.experiment-group').filter({ hasText: 'Test Group' });
    const groupSelectAll = groupDiv.locator('th.col-checkbox input[type="checkbox"]').first();

    // Check the select-all checkbox
    await groupSelectAll.check();

    // Wait a bit for state to update
    await page.waitForTimeout(100);

    // Take a screenshot to verify
    await page.screenshot({ path: 'test-results/select-all-checked.png' });

    // All checkboxes in the group should be checked
    const exp1Checkbox = page.locator('tr.main-row[data-exp-id="1"] input[type="checkbox"]');
    const exp2Checkbox = page.locator('tr.main-row[data-exp-id="2"] input[type="checkbox"]');

    await expect(exp1Checkbox).toBeChecked();
    await expect(exp2Checkbox).toBeChecked();

    // Ungrouped experiment should NOT be selected
    const exp3Checkbox = page.locator('tr.main-row[data-exp-id="3"] input[type="checkbox"]');
    await expect(exp3Checkbox).not.toBeChecked();

    // Bulk actions should appear
    await expect(page.locator('.bulk-actions')).toBeVisible();
  });

  test('duplicate experiment creates a copy', async ({ page }) => {
    // Mock the create experiment endpoint
    let createRequestBody: any = null;
    await page.route('/api/experiments', async (route) => {
      if (route.request().method() === 'POST') {
        createRequestBody = JSON.parse(route.request().postData() || '{}');
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            experiment: {
              id: '4',
              name: 'experiment-1 (copy)',
            },
          }),
        });
      }
    });

    // Mock the add to group endpoint
    await page.route('/api/groups/test-group-1/experiments', async (route) => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Mock the reorder endpoint
    await page.route('/api/groups/test-group-1/reorder', async (route) => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Select the first experiment
    const exp1Checkbox = page.locator('tr.main-row[data-exp-id="1"] input[type="checkbox"]');
    await exp1Checkbox.check();

    // Take a screenshot before clicking duplicate
    await page.screenshot({ path: 'test-results/before-duplicate.png' });

    // Click duplicate button
    await page.locator('.bulk-actions button:has-text("Duplicate")').click();

    // Wait for the duplicate to complete
    await page.waitForTimeout(500);

    // Take a screenshot after duplicate
    await page.screenshot({ path: 'test-results/after-duplicate.png' });

    // Verify the create request was made with correct data
    expect(createRequestBody).not.toBeNull();
    expect(createRequestBody.name).toBe('experiment-1 (copy)');
    expect(createRequestBody.nodes).toBe(2);
    expect(createRequestBody.gpus).toBe(4);
    expect(createRequestBody.flags).toEqual({ 'trainer.batch_size': 32, 'model.hidden_dim': 256 });

    // Should show success notification
    await expect(page.locator('.notification-success')).toBeVisible({ timeout: 5000 });
  });

  test('browser console shows correct debug info when select all is clicked', async ({ page }) => {
    // Listen to console messages
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      if (msg.text().includes('[onSelectAll]')) {
        consoleMessages.push(msg.text());
      }
    });

    // Wait for group to load
    await expect(page.locator('text=Test Group')).toBeVisible();

    // Click select all checkbox in group
    const groupDiv = page.locator('.experiment-group').filter({ hasText: 'Test Group' });
    const groupSelectAll = groupDiv.locator('th.col-checkbox input[type="checkbox"]').first();
    await groupSelectAll.check();

    // Wait for console message
    await page.waitForTimeout(500);

    // Check that debug message was logged
    expect(consoleMessages.length).toBeGreaterThan(0);

    // Log the messages for debugging
    console.log('Console messages captured:', consoleMessages);
  });
});
