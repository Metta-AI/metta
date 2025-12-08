import { test, expect } from '@playwright/test';

test.describe('SkyDeck Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses for testing
    await page.route('/api/health', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          num_experiments: 5,
          num_running_jobs: 2,
          skypilot: { staleness_seconds: 10.5 },
          s3: { staleness_seconds: 5.2 },
          observatory: { staleness_seconds: 15.3 },
        }),
      });
    });

    await page.route('/api/groups', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          groups: [],
          ungrouped: [
            {
              id: 'test-exp-1',
              name: 'Test Experiment 1',
              base_command: 'lt',
              tool_path: 'recipes.experiment.test',
              git_branch: 'main',
              nodes: 2,
              gpus: 4,
              instance_type: null,
              cloud: null,
              spot: false,
              flags: { 'trainer.batch_size': 32, 'model.hidden_dim': 256 },
              description: 'Test experiment',
              tags: [],
              group: null,
              desired_state: 'RUNNING',
              current_state: 'RUNNING',
              current_job_id: 'job-123',
              starred: false,
              is_expanded: false,
              latest_epoch: 100,
              exp_order: 0,
            },
            {
              id: 'test-exp-2',
              name: 'Test Experiment 2',
              base_command: 'lt',
              tool_path: 'recipes.experiment.test',
              git_branch: 'feature',
              nodes: 1,
              gpus: 2,
              instance_type: null,
              cloud: null,
              spot: true,
              flags: { 'trainer.batch_size': 64 },
              description: null,
              tags: [],
              group: null,
              desired_state: 'STOPPED',
              current_state: 'STOPPED',
              current_job_id: null,
              starred: true,
              is_expanded: false,
              latest_epoch: 50,
              exp_order: 1,
            },
          ],
        }),
      });
    });

    // Also mock the experiments endpoint for creating new experiments
    await page.route('/api/experiments', async route => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ experiment: { id: 'abc12345', name: 'New Experiment' } }),
        });
      } else {
        await route.fulfill({ status: 200, body: '{}' });
      }
    });

    await page.route('/api/skypilot-jobs*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jobs: [
            {
              id: 'job-123',
              experiment_id: 'test-exp-1',
              status: 'RUNNING',
              nodes: 2,
              gpus: 4,
              command: 'lt --nodes=2 --gpus=4 recipes.experiment.test run=test-exp-1',
              started_at: new Date(Date.now() - 3600000).toISOString(),
              ended_at: null,
            },
          ],
        }),
      });
    });

    await page.route('/api/settings/jobs-filters', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ value: { showStopped: false, showOrphaned: false, limit: 20 } }),
        });
      } else {
        await route.fulfill({ status: 200, body: '{}' });
      }
    });

    await page.goto('/');
  });

  test('page loads and displays health status', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle('SkyDeck Dashboard');

    // Check health status pills are displayed
    await expect(page.locator('.staleness-pill')).toHaveCount(4);
    await expect(page.locator('text=5 experiments | 2 active jobs')).toBeVisible();
  });

  test('experiments table renders with data', async ({ page }) => {
    // Check experiments are displayed (using name, not id)
    await expect(page.locator('text=Test Experiment 1')).toBeVisible();
    await expect(page.locator('text=Test Experiment 2')).toBeVisible();

    // Check status badges
    await expect(page.locator('.status-badge.running')).toBeVisible();
    await expect(page.locator('.status-badge.stopped')).toBeVisible();

    // Check epoch values
    await expect(page.locator('text=100')).toBeVisible();
    await expect(page.locator('text=50')).toBeVisible();

    // Check resources
    await expect(page.locator('text=2×4')).toBeVisible();
    await expect(page.locator('text=1×2')).toBeVisible();
  });

  test('experiment row can be expanded', async ({ page }) => {
    // Mock the jobs and checkpoints endpoints for expanded view
    await page.route('/api/experiments/test-exp-1/jobs*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jobs: [{ id: 'job-123', status: 'RUNNING', experiment_id: 'test-exp-1' }],
        }),
      });
    });

    await page.route('/api/experiments/test-exp-1/checkpoints*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          checkpoints: [
            { epoch: 100, model_path: 's3://bucket/path', version: 'v1', policy_version: '1.0', created_at: new Date().toISOString(), replay_paths: [] },
          ],
        }),
      });
    });

    await page.route('/api/experiments/test-exp-1/expanded', async route => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Click on experiment row to expand
    await page.locator('tr.main-row[data-exp-id="test-exp-1"]').click();

    // Check expanded content is visible - use more specific selectors
    await expect(page.locator('.expanded-details h3:has-text("Configuration")')).toBeVisible();
    await expect(page.locator('.expanded-details h3:has-text("Jobs")')).toBeVisible();
    await expect(page.locator('.expanded-details h3:has-text("Checkpoints")')).toBeVisible();
  });

  test('select all checkbox works', async ({ page }) => {
    // Find and click select all checkbox
    const selectAllCheckbox = page.locator('thead input[type="checkbox"]').first();
    await selectAllCheckbox.check();

    // Check that individual row checkboxes are checked
    const rowCheckboxes = page.locator('tbody tr.main-row input[type="checkbox"]');
    const count = await rowCheckboxes.count();
    expect(count).toBe(2);

    for (let i = 0; i < count; i++) {
      await expect(rowCheckboxes.nth(i)).toBeChecked();
    }

    // Check bulk actions appear
    await expect(page.locator('.bulk-actions')).toBeVisible();
  });

  test('star button toggles', async ({ page }) => {
    await page.route('/api/experiments/test-exp-1/starred', async route => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Find star button for first experiment
    const starBtn = page.locator('tr.main-row[data-exp-id="test-exp-1"] .star-btn');

    // Initially not starred
    await expect(starBtn).not.toHaveClass(/starred/);

    // Click to star
    await starBtn.click();

    // Should now be starred
    await expect(starBtn).toHaveClass(/starred/);
  });

  test('jobs table renders and filters work', async ({ page }) => {
    // Check jobs section exists
    await expect(page.locator('h2:has-text("SkyPilot Jobs")')).toBeVisible();

    // Check filter controls
    await expect(page.locator('label:has-text("My Jobs")')).toBeVisible();
    await expect(page.locator('label:has-text("Show Stopped")')).toBeVisible();
    await expect(page.locator('label:has-text("Orphaned Only")')).toBeVisible();

    // Uncheck "My Jobs" filter to see all jobs (mock job has daveey prefix but filter is strict)
    // The mock returns jobs with experiment_id starting with test-exp, not daveey.
    // So we need to uncheck the "My Jobs" filter
    await page.locator('label:has-text("My Jobs") input[type="checkbox"]').uncheck();

    // Check job is displayed
    await expect(page.locator('#jobs-table td:has-text("job-123")')).toBeVisible();
  });

  test('quick create experiment button works', async ({ page }) => {
    // Note: /api/experiments POST is already mocked in beforeEach
    // Find and click the create button
    const createBtn = page.locator('.create-btn');
    await createBtn.click();

    // Should show success notification
    await expect(page.locator('.notification-success')).toBeVisible({ timeout: 5000 });
  });

  test('scroll position is preserved during refresh', async ({ page }) => {
    // This test verifies the key issue - scroll position preservation during background refresh

    // Make the container scrollable by setting a smaller viewport height
    await page.setViewportSize({ width: 1280, height: 400 });

    // Wait for content to load
    await page.waitForSelector('tr.main-row');

    // Scroll within the experiments section
    const scrollAmount = 100;
    await page.evaluate((amount) => {
      window.scrollTo(0, amount);
    }, scrollAmount);

    // Wait a moment for scroll to settle
    await page.waitForTimeout(100);

    // Get initial scroll position
    const initialScrollY = await page.evaluate(() => window.scrollY);

    // Wait for a background refresh (5 second interval)
    await page.waitForTimeout(6000);

    // Check scroll position is preserved (within tolerance)
    const finalScrollY = await page.evaluate(() => window.scrollY);
    expect(Math.abs(finalScrollY - initialScrollY)).toBeLessThan(50);
  });

  test('state transition buttons work', async ({ page }) => {
    await page.route('/api/experiments/test-exp-1/state', async route => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Click on state cell to show edit controls
    const stateCell = page.locator('tr.main-row[data-exp-id="test-exp-1"] .col-state');
    await stateCell.click();

    // Should show stop and start buttons (check for first one)
    await expect(page.locator('tr.main-row[data-exp-id="test-exp-1"] .state-pill').first()).toBeVisible();
  });

  test('copy to clipboard works', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);

    // Click on experiment name to copy
    const expNameSpan = page.locator('tr.main-row[data-exp-id="test-exp-1"] .col-id span[title="Click to copy name"]');
    await expNameSpan.click();

    // Check clipboard content (should be the name, not the id)
    const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboardText).toBe('Test Experiment 1');
  });

  test('flag columns display correctly', async ({ page }) => {
    // Wait for experiments to load
    await page.waitForSelector('tr.main-row');

    // Check that flag values are displayed in the table
    // The flags from mock data are: trainer.batch_size (32, 64) and model.hidden_dim (256)
    // Look for these values in the table
    await expect(page.locator('td:has-text("32")')).toBeVisible();
    await expect(page.locator('td:has-text("64")')).toBeVisible();
    await expect(page.locator('td:has-text("256")')).toBeVisible();
  });

  test('notifications can be dismissed', async ({ page }) => {
    // Note: /api/experiments POST is already mocked in beforeEach
    const createBtn = page.locator('.create-btn');
    await createBtn.click();

    // Wait for notification
    const notification = page.locator('.notification-success');
    await expect(notification).toBeVisible({ timeout: 5000 });

    // Notification should auto-dismiss after 5 seconds
    await expect(notification).not.toBeVisible({ timeout: 7000 });
  });

  test('flag typeahead shows and allows adding flags', async ({ page }) => {
    // Mock the flags API endpoint
    await page.route('/api/flags*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tool_path: 'recipes.experiment.test',
          flags: [
            { flag: 'trainer.batch_size', type: 'int', default: 1024, required: false },
            { flag: 'trainer.learning_rate', type: 'float', default: 0.0003, required: false },
            { flag: 'trainer.total_timesteps', type: 'int', default: 1000000, required: false },
            { flag: 'model.hidden_dim', type: 'int', default: 512, required: false },
            { flag: 'model.num_layers', type: 'int', default: 4, required: false },
          ],
        }),
      });
    });

    // Mock the update flags endpoint
    await page.route('/api/experiments/test-exp-1/flags', async route => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    // Mock the expanded endpoints
    await page.route('/api/experiments/test-exp-1/jobs*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ jobs: [] }),
      });
    });

    await page.route('/api/experiments/test-exp-1/checkpoints*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ checkpoints: [] }),
      });
    });

    await page.route('/api/experiments/test-exp-1/expanded', async route => {
      await route.fulfill({ status: 200, body: '{}' });
    });

    await page.route('/api/experiments/test-exp-1', async route => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({ status: 200, body: '{}' });
      }
    });

    // Expand the first experiment
    await page.locator('tr.main-row[data-exp-id="test-exp-1"]').click();

    // Wait for expanded view to load
    await expect(page.locator('.expanded-details')).toBeVisible();

    // Click the edit button (pencil icon)
    const editBtn = page.locator('.expanded-details a[title="Edit configuration"]');
    await editBtn.click();

    // Wait for the flag search input to appear
    const flagInput = page.locator('input[placeholder="Type to search flags..."]');
    await expect(flagInput).toBeVisible();

    // Type to search for flags
    await flagInput.fill('learning');

    // Wait for dropdown to appear
    await expect(page.locator('div:has-text("trainer.learning_rate")')).toBeVisible();

    // Check that the dropdown shows type and default value
    await expect(page.locator('text=float')).toBeVisible();
    await expect(page.locator('text=default: 0.0003')).toBeVisible();

    // Click on a flag to add it
    await page.locator('div:has-text("trainer.learning_rate")').first().click();

    // Check that the flag was added to the table with default value
    await expect(page.locator('td:has-text("trainer.learning_rate")')).toBeVisible();
    await expect(page.locator('td:has-text("0.0003")')).toBeVisible();

    // Save the configuration
    const saveBtn = page.locator('a[title="Save changes"]');
    await saveBtn.click();

    // Should show success notification
    await expect(page.locator('.notification-success')).toBeVisible({ timeout: 5000 });
  });

  test('flag typeahead filters by search query', async ({ page }) => {
    // Mock the flags API endpoint with many flags
    await page.route('/api/flags*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tool_path: 'recipes.experiment.test',
          flags: [
            { flag: 'trainer.batch_size', type: 'int', default: 1024, required: false },
            { flag: 'trainer.learning_rate', type: 'float', default: 0.0003, required: false },
            { flag: 'model.hidden_dim', type: 'int', default: 512, required: false },
            { flag: 'model.num_layers', type: 'int', default: 4, required: false },
            { flag: 'evaluator.epoch_interval', type: 'int', default: 10, required: false },
          ],
        }),
      });
    });

    // Mock the expanded endpoints
    await page.route('/api/experiments/test-exp-1/**', async route => {
      await route.fulfill({ status: 200, body: JSON.stringify({ jobs: [], checkpoints: [] }) });
    });

    // Expand and start editing
    await page.locator('tr.main-row[data-exp-id="test-exp-1"]').click();
    await page.locator('.expanded-details a[title="Edit configuration"]').click();

    const flagInput = page.locator('input[placeholder="Type to search flags..."]');
    await expect(flagInput).toBeVisible();

    // Type "model" to filter
    await flagInput.fill('model');

    // Should only show model.* flags
    await expect(page.locator('div:has-text("model.hidden_dim")')).toBeVisible();
    await expect(page.locator('div:has-text("model.num_layers")')).toBeVisible();

    // trainer flags should not be visible
    await expect(page.locator('div:has-text("trainer.batch_size")')).not.toBeVisible();
  });

  test('flag can be deleted in edit mode', async ({ page }) => {
    // Mock necessary endpoints
    await page.route('/api/experiments/test-exp-1/**', async route => {
      await route.fulfill({ status: 200, body: JSON.stringify({ jobs: [], checkpoints: [] }) });
    });

    await page.route('/api/flags*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tool_path: 'recipes.experiment.test', flags: [] }),
      });
    });

    // Expand the first experiment
    await page.locator('tr.main-row[data-exp-id="test-exp-1"]').click();
    await expect(page.locator('.expanded-details')).toBeVisible();

    // Start editing
    await page.locator('.expanded-details a[title="Edit configuration"]').click();

    // The experiment has trainer.batch_size flag
    await expect(page.locator('td:has-text("trainer.batch_size")')).toBeVisible();

    // Click the delete button (×) next to the flag
    const deleteBtn = page.locator('button[title="Delete flag"]').first();
    await deleteBtn.click();

    // Flag should be removed from the table
    await expect(page.locator('td:has-text("trainer.batch_size")')).not.toBeVisible();
  });
});
