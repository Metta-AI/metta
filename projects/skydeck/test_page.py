"""Test script to diagnose the star button issue with Playwright."""

import asyncio

from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Navigate to the dashboard
        await page.goto("http://127.0.0.1:8000")

        # Wait for experiments to load
        await page.wait_for_selector("tr.main-row", timeout=10000)

        # Get the first experiment row
        first_row = await page.query_selector("tr.main-row")

        if first_row:
            # Get the data-exp-id attribute
            exp_id = await first_row.get_attribute("data-exp-id")
            print(f"Row data-exp-id: {exp_id}")

            # Get the ID cell content
            id_cell = await first_row.query_selector("td.col-id")
            id_text = await id_cell.text_content()
            print(f"ID cell text content: {repr(id_text)}")

            # Get the span with the actual ID
            id_span = await id_cell.query_selector("span")
            if id_span:
                span_text = await id_span.text_content()
                print(f"ID span text: {repr(span_text)}")

            # Check if there's a checkpoints container in an expanded row
            expanded_row = await page.query_selector(f'tr.expanded-row[data-exp-id="{exp_id}"]')
            if expanded_row:
                print("Expanded row exists")
                checkpoints_div = await expanded_row.query_selector(f"#checkpoints-{exp_id}")
                if checkpoints_div:
                    print(f"Found checkpoints container with ID: checkpoints-{exp_id}")
                else:
                    print(f"Checkpoints container NOT found with ID: checkpoints-{exp_id}")
                    # Try to find what ID it actually has
                    all_checkpoint_divs = await expanded_row.query_selector_all("[id^='checkpoints-']")
                    for div in all_checkpoint_divs:
                        actual_id = await div.get_attribute("id")
                        print(f"Found checkpoints container with ID: {actual_id}")

            # Click the row to expand it
            print("\nClicking row to expand...")
            await first_row.click()
            await page.wait_for_timeout(1000)  # Wait for expansion

            # Check the expanded row again
            expanded_row = await page.query_selector(f'tr.expanded-row[data-exp-id="{exp_id}"].show')
            if expanded_row:
                print("Row is now expanded")
                checkpoints_div = await expanded_row.query_selector(f"#checkpoints-{exp_id}")
                if checkpoints_div:
                    print(f"Checkpoints container found: checkpoints-{exp_id}")
                else:
                    print(f"Checkpoints container STILL not found: checkpoints-{exp_id}")
                    # Find all divs with checkpoints prefix
                    all_divs_with_id = await expanded_row.query_selector_all("[id]")
                    for div in all_divs_with_id:
                        actual_id = await div.get_attribute("id")
                        if "checkpoints" in actual_id:
                            print(f"Found: {actual_id}")

        # Keep browser open for inspection
        print("\nBrowser will stay open for 10 seconds...")
        await page.wait_for_timeout(10000)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
