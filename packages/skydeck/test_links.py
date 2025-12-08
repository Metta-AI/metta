"""Test obs and metta link functionality in SkyDeck dashboard."""

import asyncio

from playwright.async_api import async_playwright
from skydeck.services import ServiceEndpoints


async def test_links():
    """Test obs and metta links in checkpoints table."""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Grant clipboard permissions
        await context.grant_permissions(["clipboard-read", "clipboard-write"])

        page = await context.new_page()

        # Navigate to dashboard
        await page.goto("http://localhost:8000")

        # Wait for page to load
        await page.wait_for_selector("#experiments-section")
        print("✓ Dashboard loaded")

        # Find first experiment row with expand button
        expand_buttons = await page.query_selector_all(".expand-icon")
        if not expand_buttons:
            print("✗ No experiments with expand buttons found")
            await browser.close()
            return

        # Click first expand button to show checkpoints
        print("\nExpanding first experiment...")
        await expand_buttons[0].click()

        # Wait for checkpoints table to appear
        await page.wait_for_selector(".storage-pill", timeout=5000)
        await asyncio.sleep(0.5)

        # Re-query obs pills after expansion
        obs_pills = await page.query_selector_all(".storage-pill:has-text('obs')")
        if obs_pills:
            print(f"✓ Found {len(obs_pills)} obs pill(s)")

            # Click first obs pill
            print("\nTesting obs pill...")
            await obs_pills[0].click()
            await asyncio.sleep(0.5)

            # Read clipboard
            clipboard_text = await page.evaluate("navigator.clipboard.readText()")
            print(f"  Copied URL: {clipboard_text}")

            # Verify format - should be observatory web URL, not API
            expected_base = ServiceEndpoints.OBSERVATORY_WEB.replace("https://", "")
            if f"{expected_base}/policy/" in clipboard_text:
                print("  ✓ URL format is correct (uses Observatory web URL)")
                # Check if it contains a UUID (synced) or just the policy name (fallback)
                if len(clipboard_text.split("/policy/")[-1]) == 36:
                    print("  ℹ Uses synced policy_id format")
                else:
                    print("  ℹ Uses fallback format (policy name)")
            else:
                print(f"  ✗ URL format is incorrect (expected {expected_base}/policy/...)")
        else:
            print("✗ No obs pills found")

        # Find metta pill
        metta_pills = await page.query_selector_all(".storage-pill:has-text('metta')")
        if metta_pills:
            print(f"\n✓ Found {len(metta_pills)} metta pill(s)")

            # Click first metta pill
            print("\nTesting metta pill...")
            await metta_pills[0].click()
            await asyncio.sleep(0.5)

            # Read clipboard
            clipboard_text = await page.evaluate("navigator.clipboard.readText()")
            print(f"  Copied URL: {clipboard_text}")

            # Verify format
            if clipboard_text.startswith("metta://policies/") or clipboard_text.startswith("metta://policy/"):
                print("  ✓ URL format is correct (uses metta:// protocol)")
                if clipboard_text.startswith("metta://policies/"):
                    print("  ✓ Uses version format (metta://policies/version)")
                else:
                    print("  ℹ Uses fallback format (metta://policy/experiment_id)")
            else:
                print("  ✗ URL format is incorrect")
        else:
            print("✗ No metta pills found")

        print("\n=== Test Complete ===")
        print("Keeping browser open for 5 seconds for manual inspection...")
        await asyncio.sleep(5)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_links())
