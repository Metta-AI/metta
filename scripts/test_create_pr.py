#!/usr/bin/env python3
"""Quick test script to verify gitta.create_pr works with GitHub auth.

Usage:
    uv run python scripts/test_create_pr.py

This will:
1. Create a test branch
2. Make a trivial commit (add a blank line to this file)
3. Push the branch
4. Create a PR via the GitHub REST API
5. Close the PR immediately
"""

import subprocess
import sys
import time

from gitta import github as gitta
from gitta.secrets import get_github_token

REPO = "Metta-AI/metta"


def run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    # 1. Check we can get a token
    print("Checking GitHub token...")
    token = get_github_token(required=False)
    if token:
        print(f"  ✓ Got token (first 8 chars: {token[:8]}...)")
    else:
        print("  ✗ No token found! Set GITHUB_TOKEN or configure AWS secret 'github/token'")
        sys.exit(1)

    # 2. Get current branch
    current_branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    print(f"Current branch: {current_branch}")

    # 3. Create test branch
    timestamp = int(time.time())
    test_branch = f"test/pr-creation-test-{timestamp}"
    print(f"Creating test branch: {test_branch}")
    run_git("checkout", "-b", test_branch)

    try:
        # 4. Make a trivial change (add a comment to this file)
        print("Making trivial commit...")
        with open(__file__, "a") as f:
            f.write(f"\n# Test commit {timestamp}\n")

        run_git("add", __file__)
        run_git("commit", "-m", f"test: PR creation test {timestamp}")

        # 5. Push the branch
        print(f"Pushing branch {test_branch}...")
        run_git("push", "-u", "origin", test_branch)

        # 6. Create PR
        print("Creating PR via gitta.create_pr...")
        pr = gitta.create_pr(
            repo=REPO,
            title=f"[TEST] PR creation test {timestamp}",
            body="This is a test PR created by `scripts/test_create_pr.py`. Please close/delete.",
            head=test_branch,
            base=current_branch,
            token=token,
            draft=True,
        )
        pr_url = pr["html_url"]
        pr_number = pr["number"]
        print(f"  ✓ PR created: {pr_url}")

        # 7. Close the PR immediately
        print(f"Closing PR #{pr_number}...")
        import httpx

        close_resp = httpx.patch(
            f"https://api.github.com/repos/{REPO}/pulls/{pr_number}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={"state": "closed"},
            timeout=10.0,
        )
        close_resp.raise_for_status()
        print(f"  ✓ PR #{pr_number} closed")

        print("\n✅ SUCCESS: gitta.create_pr works!")

    finally:
        # Cleanup: switch back to original branch
        print(f"\nCleaning up: switching back to {current_branch}...")
        run_git("checkout", current_branch)

        # Delete local test branch
        print(f"Deleting local branch {test_branch}...")
        run_git("branch", "-D", test_branch)

        # Delete remote test branch
        print(f"Deleting remote branch {test_branch}...")
        try:
            run_git("push", "origin", "--delete", test_branch)
        except subprocess.CalledProcessError:
            print("  (remote branch may not exist or already deleted)")

        # Undo the file change
        run_git("checkout", "--", __file__)


if __name__ == "__main__":
    main()


# Test commit 1766525262
