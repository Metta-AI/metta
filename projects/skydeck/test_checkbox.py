"""Test checkbox selection features in SkyDeck dashboard."""

import asyncio

from playwright.async_api import async_playwright


async def test_checkbox_features():
    """Test checkbox selection features."""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Navigate to dashboard
        await page.goto("http://localhost:8000")

        # Wait for page to load
        await page.wait_for_selector("#experiments-table")

        print("✓ Dashboard loaded")

        # Test 1: Check if checkboxes exist
        checkboxes = await page.query_selector_all("input[type='checkbox'].row-checkbox")
        print(f"✓ Found {len(checkboxes)} experiment checkboxes")

        # Test 2: Find select-all checkbox
        select_all = await page.query_selector("#select-all")
        if select_all:
            print("✓ Found select-all checkbox")
        else:
            print("✗ Select-all checkbox NOT FOUND")

        # Test 3: Click individual checkboxes
        if len(checkboxes) > 0:
            print("\nTesting individual checkbox selection...")
            checkbox = checkboxes[0]
            await checkbox.click()
            await asyncio.sleep(0.5)

            is_checked = await checkbox.is_checked()
            print(f"  First checkbox checked: {is_checked}")

            # Check if bulk actions appear
            bulk_actions = await page.query_selector("#bulk-actions")
            if bulk_actions:
                is_visible = await bulk_actions.is_visible()
                print(f"  Bulk actions visible: {is_visible}")

            # Uncheck
            await checkbox.click()
            await asyncio.sleep(0.5)
            print("  Unchecked first checkbox")

        # Test 4: Test select-all
        if select_all:
            print("\nTesting select-all checkbox...")
            # Re-query the checkbox to avoid stale reference
            select_all_btn = await page.query_selector("#select-all")
            await select_all_btn.click()
            await asyncio.sleep(0.5)

            # Re-query checkboxes after select-all to avoid stale references
            checkboxes_after_select = await page.query_selector_all("input[type='checkbox'].row-checkbox")
            checked_count = 0
            for cb in checkboxes_after_select:
                if await cb.is_checked():
                    checked_count += 1

            print(f"  Checked count after select-all: {checked_count}/{len(checkboxes_after_select)}")

            # Test unselect-all - re-query again
            select_all_btn = await page.query_selector("#select-all")
            await select_all_btn.click()
            await asyncio.sleep(0.5)

            # Re-query checkboxes after unselect-all
            checkboxes_after_unselect = await page.query_selector_all("input[type='checkbox'].row-checkbox")
            checked_count = 0
            for cb in checkboxes_after_unselect:
                if await cb.is_checked():
                    checked_count += 1

            print(f"  Checked count after unselect-all: {checked_count}/{len(checkboxes_after_unselect)}")

        # Test 5: Check bulk action buttons
        print("\nTesting bulk action buttons...")
        # Re-query checkboxes for bulk actions test
        checkboxes_for_bulk = await page.query_selector_all("input[type='checkbox'].row-checkbox")
        if len(checkboxes_for_bulk) > 0:
            # Select one checkbox
            await checkboxes_for_bulk[0].click()
            await asyncio.sleep(0.5)

            # Check which bulk action buttons exist
            buttons = {
                "Start Selected": await page.query_selector("button:has-text('Start Selected')"),
                "Stop Selected": await page.query_selector("button:has-text('Stop Selected')"),
                "Duplicate Selected": await page.query_selector("button:has-text('Duplicate Selected')"),
                "Delete Selected": await page.query_selector("button:has-text('Delete Selected')"),
            }

            for name, button in buttons.items():
                if button:
                    is_visible = await button.is_visible()
                    print(f"  {name}: {'visible' if is_visible else 'hidden'}")
                else:
                    print(f"  {name}: NOT FOUND")

        print("\n=== Test Complete ===")
        print("Keeping browser open for 10 seconds for manual inspection...")
        await asyncio.sleep(10)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_checkbox_features())
