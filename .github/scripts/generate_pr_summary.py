#!/usr/bin/env python3
"""Generate LLM summary from PR digest data using two-phase approach."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai

# Model selection configuration
MODEL_CONFIG = {
    "phase1": "gemini-2.5-flash-preview-05-20",
    "phase2": "gemini-2.5-pro-preview-05-06",
}


def get_model_config():
    """Get model configuration from environment or use defaults."""
    phase1_model = os.getenv("PHASE1_MODEL", MODEL_CONFIG["phase1"])
    phase2_model = os.getenv("PHASE2_MODEL", MODEL_CONFIG["phase2"])

    print("Model configuration:")
    print(f"  Phase 1 (PR summaries): {phase1_model}")
    print(f"  Phase 2 (synthesis): {phase2_model}")

    return phase1_model, phase2_model


def get_pr_summary(pr: Dict[str, Any], model: genai.GenerativeModel) -> str:
    """Generate a summary for a single PR."""
    # Adjust prompt based on model capabilities
    model_name = model._model_name

    if "2.5" in model_name:
        # Models with thinking capabilities
        prompt = f"""<thinking>
Analyze this PR to understand:
1. What problem it solves
2. How it changes the codebase
3. What impact it has on developers
</thinking>

Analyze this pull request and provide a concise technical summary:

PR #{pr["number"]}: {pr["title"]}
URL: {pr["html_url"]}
Author: {pr["author"]}
Merged: {pr["merged_at"]}
Labels: {", ".join(pr.get("labels", [])) if pr.get("labels") else "None"}

Description:
{pr["body"]}

Changes (diff):
{pr["diff"]}

Please provide:
1. A one-line summary of what this PR accomplishes
2. Key technical changes (2-4 bullet points)
3. Any API changes or breaking changes
4. Impact on other developers or systems

Keep the summary focused and technical. Maximum 200 words."""
    else:
        # Standard prompt for other models
        prompt = f"""Analyze this pull request and provide a concise technical summary:

PR #{pr["number"]}: {pr["title"]}
URL: {pr["html_url"]}
Author: {pr["author"]}
Merged: {pr["merged_at"]}
Labels: {", ".join(pr.get("labels", [])) if pr.get("labels") else "None"}

Description:
{pr["body"]}

Changes (diff):
{pr["diff"]}

Please provide:
1. A one-line summary of what this PR accomplishes
2. Key technical changes (2-4 bullet points)
3. Any API changes or breaking changes
4. Impact on other developers or systems

Keep the summary focused and technical. Maximum 200 words."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing PR #{pr['number']}: {e}")
        return f"[Error summarizing PR #{pr['number']}]"


def load_pr_summary_cache() -> Dict[str, Dict[str, Any]]:
    """Load cached individual PR summaries."""
    cache_file = Path(".pr-digest-cache/pr_summaries_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
                # Include model info in cache validation
                if isinstance(cache_data, dict) and "model" in cache_data:
                    return cache_data
                # Old format - return empty to regenerate
                return {}
        except Exception as e:
            print(f"Error loading PR summary cache: {e}")
    return {}


def save_pr_summary_cache(summaries: Dict[str, Dict[str, Any]], model_name: str) -> None:
    """Save individual PR summaries to cache with model info."""
    cache_dir = Path(".pr-digest-cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "pr_summaries_cache.json"

    cache_data = {"model": model_name, "summaries": summaries}

    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"  Saved PR summary cache with {len(summaries)} entries")
    except Exception as e:
        print(f"Error saving PR summary cache: {e}")


def summarize_pr_with_cache(
    pr: Dict[str, Any], model: genai.GenerativeModel, cache: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize a single PR, using cache if available."""
    cache_key = f"{pr['number']}-{pr['merged_at']}"

    # Check cache first
    if cache_key in cache:
        print(f"  Using cached summary for PR #{pr['number']}")
        return cache[cache_key]

    # Generate new summary
    print(f"  Generating new summary for PR #{pr['number']} - {pr['title'][:60]}...")
    summary = get_pr_summary(pr, model)
    summary_data = {"number": pr["number"], "title": pr["title"], "url": pr["html_url"], "summary": summary}

    # Update cache
    cache[cache_key] = summary_data
    return summary_data


def summarize_prs_parallel(
    prs: List[Dict[str, Any]], model: genai.GenerativeModel, model_name: str, max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Summarize PRs in parallel with caching."""
    pr_summaries = []
    cache_data = load_pr_summary_cache()

    # Check if cache is for the same model
    if cache_data.get("model") != model_name:
        print(f"  Cache is for different model ({cache_data.get('model')}), regenerating all summaries")
        cached_summaries = {}
    else:
        cached_summaries = cache_data.get("summaries", {})

    updated_cache = cached_summaries.copy()

    # Separate PRs into cached and uncached
    cached_results = []
    prs_to_process = []

    for pr in prs:
        cache_key = f"{pr['number']}-{pr['merged_at']}"
        if cache_key in cached_summaries:
            print(f"  Using cached summary for PR #{pr['number']}")
            cached_results.append(cached_summaries[cache_key])
        else:
            prs_to_process.append(pr)

    print(f"  Found {len(cached_results)} cached summaries, need to generate {len(prs_to_process)} new ones")

    # Process uncached PRs in parallel
    if prs_to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PR summaries
            future_to_pr = {executor.submit(get_pr_summary, pr, model): pr for pr in prs_to_process}

            # Collect results as they complete
            for future in as_completed(future_to_pr):
                pr = future_to_pr[future]
                try:
                    summary = future.result()
                    summary_data = {
                        "number": pr["number"],
                        "title": pr["title"],
                        "url": pr["html_url"],
                        "summary": summary,
                    }
                    pr_summaries.append(summary_data)

                    # Update cache
                    cache_key = f"{pr['number']}-{pr['merged_at']}"
                    updated_cache[cache_key] = summary_data

                except Exception as e:
                    print(f"Error summarizing PR #{pr['number']}: {e}")
                    pr_summaries.append(
                        {
                            "number": pr["number"],
                            "title": pr["title"],
                            "url": pr["html_url"],
                            "summary": f"[Error summarizing PR #{pr['number']}]",
                        }
                    )

    # Combine cached and new results
    all_summaries = cached_results + pr_summaries

    # Sort by PR number to maintain order (newest first)
    all_summaries.sort(key=lambda x: x["number"], reverse=True)

    # Save updated cache if we processed new PRs
    if prs_to_process:
        save_pr_summary_cache(updated_cache, model_name)

    return all_summaries


def summarize_prs_sequential(
    prs: List[Dict[str, Any]], model: genai.GenerativeModel, model_name: str
) -> List[Dict[str, Any]]:
    """Summarize PRs sequentially with caching."""
    pr_summaries = []
    cache_data = load_pr_summary_cache()

    # Check if cache is for the same model
    if cache_data.get("model") != model_name:
        print(f"  Cache is for different model ({cache_data.get('model')}), regenerating all summaries")
        cached_summaries = {}
    else:
        cached_summaries = cache_data.get("summaries", {})

    for i, pr in enumerate(prs):
        summary_data = summarize_pr_with_cache(pr, model, cached_summaries)
        pr_summaries.append(summary_data)

    # Save updated cache
    save_pr_summary_cache(cached_summaries, model_name)

    return pr_summaries


def create_final_summary_prompt(pr_summaries: List[Dict[str, str]], start_date: str, end_date: str, days: int) -> str:
    """Create prompt for final summary from individual PR summaries."""
    prompt = f"""You have individual summaries for all PRs merged into [Metta-AI/metta](https://github.com/Metta-AI/metta)
from {start_date} to {end_date} ({days} days).

Your task is to synthesize these into a cohesive weekly summary.

IMPORTANT: Your first line MUST be:
## Summary of changes from {start_date} to {end_date}

Here are the individual PR summaries:

"""

    for pr_summary in pr_summaries:
        prompt += f"---\nPR #{pr_summary['number']}: {pr_summary['title']}\n"
        prompt += f"URL: {pr_summary['url']}\n"
        prompt += f"{pr_summary['summary']}\n\n"

    prompt += """---

Now create a structured summary with:

1. **Executive Summary:**
   A concise paragraph summarizing the week's key technical developments and their overall impact.

2. **Internal API Changes:**
   List any API changes mentioned in the PR summaries, with PR references like [#123](https://github.com/Metta-AI/metta/pull/123).

3. **Detailed Breakdown:**
   Group related PRs by feature, component, or theme. Reference PRs like [#123](https://github.com/Metta-AI/metta/pull/123).

Focus on technical accuracy and create a well-structured Markdown summary for Discord."""

    return prompt


def main():
    """Generate summary from PR digest using two-phase approach."""
    print("Starting two-phase PR summarization...")

    # Read PR digest
    digest_file = Path(os.getenv("PR_DIGEST_FILE", "pr_digest_output.json"))
    if not digest_file.exists():
        print(f"Error: Digest file not found: {digest_file}")
        sys.exit(1)

    with open(digest_file) as f:
        prs = json.load(f)

    if not prs:
        print("No PRs to summarize")
        return

    # Get date range
    date_range = os.getenv("DATE_RANGE", "")
    if date_range:
        start_date_str, end_date_str = date_range.split(" to ")
    else:
        merged_dates = [pr["merged_at"] for pr in prs]
        start_date_str = min(merged_dates)[:10]
        end_date_str = max(merged_dates)[:10]

    days_scanned = int(os.getenv("DAYS_TO_SCAN", "7"))

    # Initialize Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    genai.configure(api_key=api_key)

    # Get model configuration
    phase1_model_name, phase2_model_name = get_model_config()

    # Model for individual PR summaries
    pr_model = genai.GenerativeModel(phase1_model_name)

    # Model for final summary
    system_prompt = """You are a technical writing assistant specialized in summarizing software development activity.
You synthesize individual PR summaries into structured, technical, and concise weekly reports suitable for Discord.
Focus exclusively on technical accuracy, clarity, and readability."""

    final_model = genai.GenerativeModel(phase2_model_name, system_instruction=system_prompt)

    # Phase 1: Generate individual PR summaries
    print(f"\nPhase 1: Generating summaries for {len(prs)} PRs...")

    # Check if we should use parallel processing
    use_parallel = os.getenv("USE_PARALLEL", "true").lower() == "true"
    max_workers = int(os.getenv("MAX_WORKERS", "5"))

    if use_parallel and len(prs) > 2:
        print(f"  Using parallel processing with {max_workers} workers")
        pr_summaries = summarize_prs_parallel(prs, pr_model, phase1_model_name, max_workers)
    else:
        print("  Using sequential processing")
        pr_summaries = summarize_prs_sequential(prs, pr_model, phase1_model_name)

    # Save intermediate summaries (useful for debugging)
    with open("pr_summaries_intermediate.json", "w") as f:
        json.dump(pr_summaries, f, indent=2)
    print("  Saved intermediate summaries to pr_summaries_intermediate.json")

    # Phase 2: Generate final summary
    print("\nPhase 2: Generating final consolidated summary...")
    final_prompt = create_final_summary_prompt(pr_summaries, start_date_str, end_date_str, days_scanned)

    try:
        response = final_model.generate_content(final_prompt)
        summary = response.text

        # Clean up response
        if summary.startswith("```markdown"):
            summary = summary[len("```markdown") :].lstrip()
        if summary.endswith("```"):
            summary = summary[: -len("```")].rstrip()

    except Exception as e:
        print(f"Error generating final summary: {e}")
        sys.exit(1)

    # Output
    print("\n--- Generated PR Summary ---")
    print(summary)
    print("--- End of Summary ---")

    # Save for workflow
    with open("pr_summary_output.txt", "w") as f:
        f.write(summary)

    # GitHub output
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("summary_file=pr_summary_output.txt\n")

    print("\nScript finished successfully.")


if __name__ == "__main__":
    main()
