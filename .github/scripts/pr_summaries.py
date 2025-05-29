from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import google.generativeai as genai
import requests


@dataclass(slots=True)
class PullRequestDigest:
    """
    Minimal information for an LLM-friendly summary of a merged PR.
    """

    number: int
    title: str
    body: str  # main PR description (Markdown)
    merged_at: str  # ISO-8601 timestamp
    html_url: str
    diff: str  # full unified diff or a truncated prefix


GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "Metta-AI/metta"
DEFAULT_DAYS = 7
DEFAULT_DIFF_LIMIT = 20_000

DISCORD_MESSAGE_CHARACTER_LIMIT = 2000
CACHE_FILE = "github_data_cache.json"


def _auth_header(token: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _search_merged_pr_numbers(repo: str, since: str, token: Optional[str]) -> List[int]:
    """Return PR numbers merged on/after `since` (YYYY-MM-DD)."""
    url = f"{GITHUB_API}/search/issues"
    q = f"repo:{repo} is:pr is:merged merged:>={since}"
    all_numbers: List[int] = []
    page = 1
    print(f"Searching for PRs in {repo} merged since {since}...")
    while True:
        params = {"q": q, "per_page": 100, "page": page}
        r = requests.get(url, params=params, headers=_auth_header(token), timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        numbers = [item["number"] for item in items]
        all_numbers.extend(numbers)

        if len(items) < 100:
            break
        page += 1
    print(f"Found {len(all_numbers)} merged PRs.")
    return all_numbers


def _pull_request_digest(
    repo: str,
    number: int,
    token: Optional[str],
    diff_bytes: Optional[int],
) -> PullRequestDigest:
    """Fetch body + diff for one PR."""
    pr_url = f"{GITHUB_API}/repos/{repo}/pulls/{number}"
    pr_response = requests.get(pr_url, headers=_auth_header(token), timeout=30)
    pr_response.raise_for_status()
    pr = pr_response.json()

    diff_url: str = pr["diff_url"]
    diff_headers = {"Accept": "application/vnd.github.v3.diff", **_auth_header(token)}
    diff_resp = requests.get(diff_url, headers=diff_headers, timeout=30)
    diff_resp.raise_for_status()
    diff_text = diff_resp.text
    if diff_bytes is not None and len(diff_text.encode("utf-8")) > diff_bytes:  # Check byte length
        # Truncate by bytes, ensuring valid UTF-8
        encoded_diff = diff_text.encode("utf-8")[:diff_bytes]
        try:
            diff_text = encoded_diff.decode("utf-8") + "\n…[truncated]"
        except UnicodeDecodeError:
            # If decoding fails, find the last valid UTF-8 sequence
            diff_text = encoded_diff.rpartition(b"\n")[0].decode("utf-8", "ignore") + "\n…[truncated]"

    return PullRequestDigest(
        number=number,
        title=pr["title"],
        body=pr.get("body") or "",  # Ensure body is not None
        merged_at=pr["merged_at"],
        html_url=pr["html_url"],
        diff=diff_text,
    )


def recent_merged_pr_digests(
    *,
    repo: str = DEFAULT_REPO,
    days: int = DEFAULT_DAYS,
    diff_limit: Optional[int] = DEFAULT_DIFF_LIMIT,
) -> List[PullRequestDigest]:
    """
    Return digests for every PR merged into `repo` within the last `days`.
    Assumes GITHUB_TOKEN environment variable is set for authentication.
    Caches results to CACHE_FILE if it doesn't exist, otherwise reads from cache.
    """
    # Check for cache first
    if os.path.exists(CACHE_FILE):
        print(f"Cache file '{CACHE_FILE}' found. Loading data from cache.")
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Reconstruct PullRequestDigest objects
            return [PullRequestDigest(**pr_data) for pr_data in cached_data]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error reading cache file '{CACHE_FILE}': {e}. Fetching from API.")
            # Fall through to fetching from API if cache is invalid

    effective_token = os.getenv("GITHUB_TOKEN")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    numbers = _search_merged_pr_numbers(repo, since, effective_token)
    if not numbers:
        print(f"No PRs found for {repo} merged since {since}.")
        return []

    print(f"Fetching details for {len(numbers)} PRs...")
    digests: List[PullRequestDigest] = []
    for i, n in enumerate(numbers):
        print(f"Fetching PR {i + 1}/{len(numbers)}: #{n}")
        try:
            digests.append(_pull_request_digest(repo, n, effective_token, diff_limit))
        except Exception as e:
            print(f"Error fetching PR #{n}: {e}. Skipping this PR.")

    # Save to cache
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            # Convert PullRequestDigest objects to dicts for JSON serialization
            json.dump([asdict(pr) for pr in digests], f, indent=2)
        print(f"Successfully saved PR data to cache file '{CACHE_FILE}'.")
    except IOError as e:
        print(f"Error saving PR data to cache file '{CACHE_FILE}': {e}")
    return digests


def create_summary_prompt_from_digests(prs: List[PullRequestDigest]) -> str:
    """Creates a prompt for the LLM based on PullRequestDigest objects."""
    prompt_text = "Here are the PRs merged in the last week:\n\n"

    for pr in prs:
        prompt_text += f"## PR #{pr.number}: {pr.title}\\n"
        prompt_text += f"URL: {pr.html_url}\\n"
        prompt_text += f"Merged At: {pr.merged_at}\\n\\n"
        prompt_text += "### Body:\\n"
        prompt_text += f"{pr.body[:1500]}...\\n\\n"  # Increased body limit
        prompt_text += "### Diff Preview:\\n"
        prompt_text += f"{pr.diff[:3000]}...\\n\\n"  # Increased diff limit
        prompt_text += "---\\n\\n"

    return prompt_text


def get_llm_summary(
    prompt: str,
    start_date_str: str,
    end_date_str: str,
    model_name: str = "gemini-2.5-flash-preview-04-17",
) -> str:
    """Gets a summary from the specified Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    system_prompt = """
You are a technical writing assistant specialized in summarizing software development activity. You receive detailed information about pull requests (PRs) merged into the Metta-AI/metta GitHub repository and create structured, technical, and concise Markdown summaries suitable for direct posting to Discord. 

Always adhere closely to the formatting and linking guidelines provided in the user prompt. Focus exclusively on technical accuracy, clarity, and readability.
    """

    # Construct the instruction for the first line dynamically
    first_line_instruction = (
        f"Your first line of output MUST be the following Markdown heading, using the provided dates: "
        f"## Summary of changes from {start_date_str} to {end_date_str}\\n\\n"
    )

    prompt_prefix = f"""{first_line_instruction}
You will receive a list of all pull requests (PRs) merged into the repository [Metta-AI/metta](https://github.com/Metta-AI/metta) in the past week. Each PR entry contains the title, description, and diff of changes.

Create a structured summary using Markdown, clearly highlighting the important technical changes made during the week:

1. **Executive Summary:**  
   A concise paragraph summarizing the week's key technical developments and their overall impact.

2. **Internal API Changes:**  
   If there were changes to internal APIs, explicitly list and explain these here, clearly indicating the impact on other developers. Always include PR references in brackets like [#123](https://github.com/Metta-AI/metta/pull/123).

3. **Detailed Breakdown:**  
   Group related PRs by feature, component, or theme. Clearly explain what changed, why it matters, and mention notable PRs using linked references ([#PR_Number](https://github.com/Metta-AI/metta/pull/PR_Number)). You do **not** need to mention every PR—skip trivial changes.

Use Markdown extensively (headings, bullet points, code blocks) to create a structured and easily readable summary. Your output will be directly posted into Discord, so do NOT include any additional text or explanations outside the requested structure.
    """

    try:
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        print(f"Calling LLM ({model_name}) to generate summary...")
        response = model.generate_content(prompt_prefix + prompt)
        response_text = response.text
        if response_text.startswith("```markdown"):
            response_text = response_text[len("```markdown") :].lstrip()
        if response_text.endswith("```"):
            response_text = response_text[: -len("```")].rstrip()
        return response_text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        sys.exit(1)


def main():
    """Main function to fetch PRs and generate summary."""
    print("Starting PR summarization script...")

    # Configuration (can be replaced with argparse or environment variables later)
    github_repo = os.getenv("GITHUB_REPO", DEFAULT_REPO)
    days_to_scan = int(os.getenv("DAYS_TO_SCAN", DEFAULT_DAYS))
    # Diff limit in bytes, None for no limit
    diff_character_limit = os.getenv("DIFF_CHARACTER_LIMIT")
    if diff_character_limit is not None:
        try:
            diff_character_limit = int(diff_character_limit)
        except ValueError:
            print(
                f"Warning: Invalid DIFF_CHARACTER_LIMIT '{diff_character_limit}', using default {DEFAULT_DIFF_LIMIT} characters."
            )
            diff_character_limit = DEFAULT_DIFF_LIMIT
    else:
        diff_character_limit = DEFAULT_DIFF_LIMIT
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    # Calculate date range for the summary
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_to_scan)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # 1. Fetch PR digests
    pr_digests = recent_merged_pr_digests(
        repo=github_repo,
        days=days_to_scan,
        diff_limit=diff_character_limit,
    )

    if not pr_digests:
        print("No pull requests found to summarize.")
        return

    # 2. Create prompt
    print("Creating prompt for LLM...")
    summary_prompt = create_summary_prompt_from_digests(pr_digests)
    # print(f"DEBUG: Prompt generated:\\n{summary_prompt[:500]}...") # For debugging

    # 3. Get LLM summary
    summary = get_llm_summary(summary_prompt, start_date_str, end_date_str)

    # 4. Print summary
    print("\n--- Generated PR Summary ---")
    print(summary)
    print("--- End of Summary ---")

    # 5. Send to Discord if URL is configured
    if discord_webhook_url:
        print("Sending summary to Discord...")
        send_to_discord(discord_webhook_url, summary)
    else:
        print("DISCORD_WEBHOOK_URL not set. Skipping sending to Discord.")

    print("Script finished successfully.")


def send_to_discord(webhook_url: str, content: str) -> None:
    """Sends content to a Discord webhook, handling message splitting."""
    if not webhook_url:
        print("Discord webhook URL not provided. Skipping sending message to Discord.")
        return

    max_len = DISCORD_MESSAGE_CHARACTER_LIMIT
    chunks: List[str] = []
    remaining_content = content.strip()

    while remaining_content:
        if len(remaining_content) <= max_len:
            chunks.append(remaining_content)
            break

        # Try to find a double newline (paragraph break) to split
        split_at = remaining_content.rfind("\\n\\n", 0, max_len)

        if split_at != -1:
            # Split after the double newline
            chunk = remaining_content[: split_at + 2]
            remaining_content = remaining_content[split_at + 2 :]
        else:
            # No double newline, try to find a single newline
            split_at = remaining_content.rfind("\\n", 0, max_len)
            if split_at != -1:
                # Split after the single newline
                chunk = remaining_content[: split_at + 1]
                remaining_content = remaining_content[split_at + 1 :]
            else:
                # No newline found, hard split (worst case)
                chunk = remaining_content[:max_len]
                remaining_content = remaining_content[max_len:]

        chunks.append(chunk.strip())  # Remove any leading/trailing whitespace from the new chunk

    for i, chunk in enumerate(chunks):
        if not chunk:  # Skip sending empty chunks
            continue
        payload = {"content": chunk, "flags": 4}  # flags: 4 for SUPPRESS_EMBEDS
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"Successfully sent chunk {i + 1}/{len(chunks)} to Discord.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to Discord: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Discord API response: {e.response.text}")
            break  # Stop sending further chunks if one fails


if __name__ == "__main__":
    main()
