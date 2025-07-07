import { test, expect } from '@playwright/test';

function encodeReplayUrl(url: string): string {
  return encodeURIComponent(url);
}

test('smoke test', async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  await page.goto('http://localhost:2000');
  expect(consoleErrors).toHaveLength(0);
});

test('load a replay', async ({ page }) => {
  const consoleErrors: string[] = [];
  const replayUrl = 'https://softmax-public.s3.amazonaws.com/replays/daveey.arena.baseline.8x4/3445942f5832/bd5a49ae-e8ed-47a0-a210-8452d28f8e03.json.z';
  await page.goto(`http://localhost:2000/?replayUrl=${encodeReplayUrl(replayUrl)}`);
  
  // Wait for the page to fully load the replay and render the first frame
  console.log('Waiting for replay to be ready...');
  await page.waitForFunction(() => {
    const state = (window as any).state;
    return state && state.replay !== null;
  }, { timeout: 10000 });
  expect(consoleErrors).toHaveLength(0);
});


test('load a replay and play it', async ({ page }) => {
  const consoleErrors: string[] = [];
  const replayUrl = 'https://softmax-public.s3.amazonaws.com/replays/daveey.arena.baseline.8x4/3445942f5832/bd5a49ae-e8ed-47a0-a210-8452d28f8e03.json.z';
  await page.goto(`http://localhost:2000/?replayUrl=${encodeReplayUrl(replayUrl)}&play=true`);
  
  // Wait for the page to fully load the replay and render the first frame
  console.log('Waiting for replay to be ready...');
  await page.waitForFunction(() => {
    const state = (window as any).state;
    return state && state.isPlaying == true;
  }, { timeout: 10000 });
  expect(consoleErrors).toHaveLength(0);
});
