const playwright = require('playwright');

(async () => {
  const browser = await playwright.chromium.launch({ headless: false });
  const page = await browser.newPage();

  await page.goto('http://localhost:8000');

  // Wait for experiments to load
  await page.waitForSelector('.main-row', { timeout: 10000 });

  console.log('Page loaded, waiting 2 seconds...');
  await page.waitForTimeout(2000);

  // Click to expand first experiment
  const firstRow = await page.$('.main-row');
  await firstRow.click();

  console.log('Expanded row, waiting for content...');
  await page.waitForSelector('.expanded-row.show', { timeout: 5000 });
  await page.waitForTimeout(1000);

  // Click the Edit button
  const editBtn = await page.$('[id^="config-edit-btn-"]');
  if (!editBtn) {
    console.error('Edit button not found!');
    await browser.close();
    return;
  }

  await editBtn.click();
  console.log('Clicked Edit button');

  // Check if button changed to Save
  await page.waitForTimeout(500);
  const btnText = await editBtn.textContent();
  console.log('Button text after click:', btnText);

  // Check if fields are editable
  const editableFields = await page.$$('[contenteditable="true"]');
  console.log('Number of editable fields:', editableFields.length);

  // Monitor for 10 seconds
  for (let i = 1; i <= 10; i++) {
    await page.waitForTimeout(1000);

    // Check if still in edit mode
    const currentBtnText = await editBtn.textContent();
    const currentEditableCount = (await page.$$('[contenteditable="true"]')).length;

    console.log(`After ${i}s: button="${currentBtnText}", editable fields=${currentEditableCount}`);

    if (currentBtnText !== 'Save') {
      console.error(`PROBLEM: Button reverted to "${currentBtnText}" after ${i} seconds!`);
      break;
    }

    if (currentEditableCount === 0) {
      console.error(`PROBLEM: Editable fields disappeared after ${i} seconds!`);
      break;
    }
  }

  console.log('Test complete');
  await browser.close();
})();
