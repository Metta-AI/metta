#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-generativeai>=0.3.0",
#   "requests>=2.31.0",
# ]
# ///

import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

# Model selection configuration
MODEL_CONFIG = {
    "phase1": "gemini-2.5-flash-preview-05-20",  # For individual PR analysis
    "phase2": "gemini-2.5-pro-preview-06-05",  # For collection summaries
}


@dataclass
class PRSummary:
    """Individual PR summary with essential metadata."""

    pr_number: int
    title: str
    author: str
    merged_at: str
    html_url: str
    summary: str
    key_changes: list[str]
    developer_impact: str
    technical_notes: str
    impact_level: str  # "minor", "moderate", "major"
    category: str  # "feature", "bugfix", "docs", "refactor", etc.
    artifact_path: str


class GeminiAIClient:
    """AI client optimized for Gemini 2.5 with rate limiting and retry logic."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_delay = 0.3  # Fast model, minimal delay
        self.max_retries = 3
        self.last_request_time = 0

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Minimal safety settings for code analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logging.info("Initialized AI client with Gemini 2.5 models")

    def _get_model(self, phase: str):
        """Get the appropriate model for the analysis phase."""
        model_name = MODEL_CONFIG.get(phase, MODEL_CONFIG["phase1"])

        generation_config = genai.GenerationConfig(
            temperature=0.2,  # Consistent technical analysis
            top_p=0.8,
            top_k=40,
            max_output_tokens=5000 if phase == "phase1" else 10000,
            response_mime_type="text/plain",
        )

        return genai.GenerativeModel(
            model_name=model_name, generation_config=generation_config, safety_settings=self.safety_settings
        )

    def _wait_for_rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def generate_with_retry(self, prompt: str, phase: str = "phase1") -> str | None:
        """Generate content with retry logic."""
        model = self._get_model(phase)

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = model.generate_content(prompt)

                if response.text:
                    logging.info(f"AI generation successful on attempt {attempt + 1} ({MODEL_CONFIG[phase]})")
                    return response.text.strip()
                else:
                    logging.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)
                        continue

            except Exception as e:
                logging.error(f"AI generation error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logging.error("Max retries exceeded for AI generation")
                    return None

        return None


class PRSummaryGenerator:
    """Enhanced PR summary generator with caching and concurrency."""

    def __init__(self, api_key: str, output_dir: Path = Path("pr-summaries")):
        self.ai_client = GeminiAIClient(api_key)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def categorize_pr(self, pr_data: dict) -> str:
        """Enhanced PR categorization."""
        title = pr_data.get("title", "").lower()
        labels = [label.lower() for label in pr_data.get("labels", [])]
        body = pr_data.get("body", "").lower()

        # Check labels first (most reliable)
        if any(label in ["bug", "bugfix", "fix"] for label in labels):
            return "bugfix"
        elif any(label in ["feature", "enhancement"] for label in labels):
            return "feature"
        elif any(label in ["docs", "documentation"] for label in labels):
            return "documentation"
        elif any(label in ["refactor", "cleanup"] for label in labels):
            return "refactor"
        elif any(label in ["test", "testing"] for label in labels):
            return "testing"
        elif any(label in ["ci", "build", "deployment"] for label in labels):
            return "infrastructure"
        elif any(label in ["security"] for label in labels):
            return "security"

        # Check title patterns
        if any(word in title for word in ["fix", "bug", "error", "resolve"]):
            return "bugfix"
        elif any(word in title for word in ["add", "new", "feature", "implement"]):
            return "feature"
        elif any(word in title for word in ["doc", "readme", "guide"]):
            return "documentation"
        elif any(word in title for word in ["refactor", "clean", "improve", "optimize"]):
            return "refactor"
        elif any(word in title for word in ["test", "spec", "coverage"]):
            return "testing"
        elif any(word in title for word in ["bump", "update", "upgrade", "dependency"]):
            return "dependency"
        elif any(word in title for word in ["security", "vulnerability", "cve"]):
            return "security"

        # Check body for additional context
        if any(word in body for word in ["fixes #", "closes #", "resolves #"]):
            return "bugfix"

        return "other"

    def estimate_impact_level(self, pr_data: dict) -> str:
        """Enhanced impact estimation."""
        diff = pr_data.get("diff", "")
        title = pr_data.get("title", "").lower()
        labels = [label.lower() for label in pr_data.get("labels", [])]

        lines_added = diff.count("\n+")
        lines_removed = diff.count("\n-")
        total_lines = lines_added + lines_removed

        files_changed = len(re.findall(r"^diff --git", diff, re.MULTILINE))

        # Check for breaking changes indicators
        breaking_indicators = ["breaking", "major", "api change", "migration"]
        if any(indicator in title for indicator in breaking_indicators):
            return "major"
        if any(indicator in labels for indicator in breaking_indicators):
            return "major"

        # Check for critical patterns in diff
        critical_patterns = [
            r"class\s+\w+.*:",  # New classes
            r"def\s+__init__",  # Constructor changes
            r"@api\.|@endpoint",  # API decorators
            r"DELETE FROM|DROP TABLE|ALTER TABLE",  # Database changes
        ]

        if any(re.search(pattern, diff, re.IGNORECASE) for pattern in critical_patterns):
            if total_lines > 100:
                return "major"
            else:
                return "moderate"

        # Size-based classification
        if total_lines > 500 or files_changed > 10:
            return "major"
        elif total_lines > 100 or files_changed > 5:
            return "moderate"
        else:
            return "minor"

    def load_cache(self) -> dict:
        """Load cached PR summaries."""
        force_refresh = os.getenv("FORCE_REFRESH", "false").lower() == "true"
        if force_refresh:
            logging.info("Force refresh enabled, skipping cache load")
            return {}

        cache_file = Path(".pr-digest-cache/enhanced_pr_summaries_cache.json")
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cache_data = json.load(f)
                    if isinstance(cache_data, dict) and "model" in cache_data:
                        return cache_data
                return {}
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
        return {}

    def save_cache(self, summaries: dict, model_name: str) -> None:
        """Save PR summaries to cache."""
        cache_dir = Path(".pr-digest-cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "enhanced_pr_summaries_cache.json"

        cache_data = {"model": model_name, "generated_at": datetime.now().isoformat(), "summaries": summaries}

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logging.info(f"Saved cache with {len(summaries)} entries")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def generate_single_pr_summary(self, pr_data: dict) -> dict | None:
        """Generate comprehensive summary for a single PR."""
        pr_number = pr_data["number"]

        # Use full context with token estimation
        full_diff = pr_data.get("diff", "")
        full_description = pr_data.get("body", "No description provided")

        estimated_tokens = len(full_diff + full_description) // 4
        logging.info(f"PR #{pr_number}: Using ~{estimated_tokens:,} tokens")

        prompt = self._create_pr_analysis_prompt(pr_data, full_diff, full_description)

        # Generate with phase1 model
        ai_response = self.ai_client.generate_with_retry(prompt, "phase1")

        if not ai_response:
            logging.error(f"AI generation failed for PR #{pr_number}")
            return None

        # Parse AI response
        parsed = self._parse_ai_response(ai_response)

        # Get metadata
        category = self.categorize_pr(pr_data)
        impact_level = self.estimate_impact_level(pr_data)

        # Create artifact
        artifact_path = self._create_artifact(pr_data, parsed, category, impact_level)

        return {
            "pr_number": pr_number,
            "title": pr_data["title"],
            "author": pr_data["author"],
            "merged_at": pr_data["merged_at"],
            "html_url": pr_data["html_url"],
            "summary": parsed["summary"],
            "key_changes": parsed["key_changes"],
            "developer_impact": parsed["developer_impact"],
            "technical_notes": parsed["technical_notes"],
            "impact_level": impact_level,
            "category": category,
            "artifact_path": artifact_path,
        }

    def _create_pr_analysis_prompt(self, pr_data: dict, full_diff: str, full_description: str) -> str:
        # Extract file info for context
        file_changes = re.findall(r"^diff --git a/(.+?) b/(.+?)$", full_diff, re.MULTILINE)
        file_types = set()
        for _old_file, new_file in file_changes:
            if "." in new_file:
                file_types.add(new_file.split(".")[-1])

        lines_added = full_diff.count("\n+")
        lines_removed = full_diff.count("\n-")

        prompt = f"""
Analyze this Pull Request comprehensively for an engineering audience.

**PR #{pr_data["number"]}: {pr_data["title"]}**
- Author: {pr_data["author"]}
- Labels: {", ".join(pr_data.get("labels", [])) if pr_data.get("labels") else "None"}
- Files: {len(file_changes)} files, types: {", ".join(sorted(file_types))}
- Changes: {lines_added} additions, {lines_removed} deletions

**Complete Description:**
{full_description}

**Complete Code Changes:**
{full_diff}

**Required Output Format:**

**Summary:** [2-3 sentences explaining what this PR accomplishes, why it was needed, and its primary benefit]

**Key Changes:**
• [Most significant functional or architectural change with specific details]
• [Second most important change - focus on implementation or API changes]
• [Third change if applicable - additional notable modifications]

**Developer Impact:** [Concrete effects on other developers: new/changed APIs, breaking changes, new dependencies,
modified workflows, testing requirements. If truly minimal, state "Minimal developer impact."]

**Technical Notes:** [Implementation details, architectural decisions, performance implications, trade-offs, or
gotchas worth highlighting]

**Analysis Guidelines:**
- Use the COMPLETE diff and description - analyze actual code changes, not just intentions
- Focus on concrete, measurable changes rather than abstract descriptions
- Prioritize information that helps engineers understand integration points and potential conflicts
- Highlight any non-obvious implications or technical debt considerations
- Be precise about scope: distinguish between refactoring, new features, and bug fixes

Maximum 500 words total. Be comprehensive within this limit."""
        return prompt

    def _parse_ai_response(self, response: str) -> dict:
        """Parse AI response into structured data."""
        parsed = {"summary": "", "key_changes": [], "developer_impact": "", "technical_notes": ""}

        # Extract summary
        summary_match = re.search(r"\*\*Summary:\*\*\s*(.+?)(?=\*\*|$)", response, re.DOTALL)
        if summary_match:
            parsed["summary"] = summary_match.group(1).strip()

        # Extract key changes
        changes_section = re.search(r"\*\*Key Changes:\*\*\s*((?:•.*?(?:\n|$))+)", response, re.DOTALL)
        if changes_section:
            changes_text = changes_section.group(1)
            parsed["key_changes"] = [
                change.strip().lstrip("•").strip()
                for change in changes_text.split("\n")
                if change.strip() and "•" in change
            ]

        # Extract developer impact
        impact_match = re.search(r"\*\*Developer Impact:\*\*\s*(.+?)(?=\*\*|$)", response, re.DOTALL)
        if impact_match:
            parsed["developer_impact"] = impact_match.group(1).strip()

        # Extract technical notes
        notes_match = re.search(r"\*\*Technical Notes:\*\*\s*(.+?)(?=\*\*|$)", response, re.DOTALL)
        if notes_match:
            parsed["technical_notes"] = notes_match.group(1).strip()

        return parsed

    def _create_artifact(self, pr_data: dict, parsed_summary: dict, category: str, impact_level: str) -> str:
        """Create detailed artifact file."""
        pr_number = pr_data["number"]
        artifact_filename = f"pr-{pr_number}-summary.md"
        artifact_path = self.output_dir / artifact_filename

        content = f"""# PR #{pr_number}: {pr_data["title"]}

**Author:** {pr_data["author"]}
**Merged:** {pr_data["merged_at"]}
**Category:** {category.title()}
**Impact Level:** {impact_level.title()}
**GitHub URL:** {pr_data["html_url"]}

## Summary

{parsed_summary["summary"]}

## Key Changes

{chr(10).join(f"- {change}" for change in parsed_summary["key_changes"])}

## Developer Impact

{parsed_summary["developer_impact"]}

## Technical Notes

{parsed_summary.get("technical_notes", "No additional technical notes")}

## Original Description

{pr_data.get("body", "No description provided")}

## Labels

{", ".join(pr_data.get("labels", [])) if pr_data.get("labels") else "None"}

<details>
<summary>Complete Code Diff (Click to expand)</summary>

```diff
{pr_data.get("diff", "No diff available")}
```

</details>

---
*Generated using {MODEL_CONFIG["phase1"]} on {datetime.now().isoformat()}*
"""

        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(artifact_path)

    def process_prs_with_cache(self, prs: list, use_parallel: bool = True, max_workers: int = 5) -> list[PRSummary]:
        """Process PRs with caching and optional concurrency."""

        # Load cache
        cache_data = self.load_cache()
        model_name = MODEL_CONFIG["phase1"]

        if cache_data.get("model") != model_name:
            logging.info(f"Cache is for different model ({cache_data.get('model')}), regenerating")
            cached_summaries = {}
        else:
            cached_summaries = cache_data.get("summaries", {})

        updated_cache = cached_summaries.copy()
        results = []
        prs_to_process = []

        # Separate cached and uncached PRs
        for pr in prs:
            cache_key = f"{pr['number']}-{pr['merged_at']}"
            if cache_key in cached_summaries:
                logging.info(f"Using cached summary for PR #{pr['number']}")
                results.append(PRSummary(**cached_summaries[cache_key]))
            else:
                prs_to_process.append(pr)

        logging.info(f"Found {len(results)} cached summaries, processing {len(prs_to_process)} new ones")

        # Process uncached PRs
        if prs_to_process:
            if use_parallel and len(prs_to_process) > 2:
                logging.info(f"Using parallel processing with {max_workers} workers")
                new_summaries = self._process_parallel(prs_to_process, max_workers)
            else:
                logging.info("Using sequential processing")
                new_summaries = self._process_sequential(prs_to_process)

            # Add to results and cache
            for summary_data in new_summaries:
                if summary_data:
                    summary = PRSummary(**summary_data)
                    results.append(summary)

                    cache_key = f"{summary.pr_number}-{summary.merged_at}"
                    updated_cache[cache_key] = summary_data

            # Save updated cache
            self.save_cache(updated_cache, model_name)

        # Sort by PR number (newest first)
        results.sort(key=lambda x: x.pr_number, reverse=True)
        return results

    def _process_parallel(self, prs: list, max_workers: int) -> list:
        """Process PRs in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pr = {executor.submit(self.generate_single_pr_summary, pr): pr for pr in prs}

            for future in as_completed(future_to_pr):
                pr = future_to_pr[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logging.info(f"Completed PR #{pr['number']}")
                    else:
                        logging.error(f"Failed to process PR #{pr['number']}")
                except Exception as e:
                    logging.error(f"Error processing PR #{pr['number']}: {e}")

        return results

    def _process_sequential(self, prs: list) -> list:
        """Process PRs sequentially."""
        results = []

        for i, pr in enumerate(prs):
            logging.info(f"Processing PR {i + 1}/{len(prs)}: #{pr['number']}")
            try:
                result = self.generate_single_pr_summary(pr)
                if result:
                    results.append(result)
                else:
                    logging.error(f"Failed to process PR #{pr['number']}")
            except Exception as e:
                logging.error(f"Error processing PR #{pr['number']}: {e}")

        return results

    def generate_collection_summary(self, pr_summaries: list[PRSummary], date_range: str, repository: str) -> str:
        """Generate collection summary and shout outs using Gemini 2.5 Pro."""

        context = self._prepare_collection_context(pr_summaries, date_range, repository)

        bonus_prompts = [
            "A haiku capturing the essence of this development cycle - focus on the rhythm of progress, "
            "bugs conquered, or features born",
            "A team affirmation based on the collective accomplishments and growth demonstrated in this period's work",
            "A zen koan that emerges from the deepest technical challenge or breakthrough of this period - make it "
            "thought-provoking for engineers",
        ]

        selected_bonus_prompt = random.choice(bonus_prompts)

        prompt = f"""You are creating an executive summary of development activity for {repository} from {date_range}.

INPUT DATA: Below you'll find PR summaries with titles, descriptions, authors, file changes, and technical details:

{context}

AUDIENCE & TONE: Technical and direct - written for engineering and research staff who understand code architecture,
implementation details, and technical tradeoffs.

QUALITY CRITERIA:
- Focus on technical impact and engineering outcomes
  • Good: "Reduced API response time by 40% through query optimization and caching"
  • Good: "Eliminated memory leaks in background workers, improving system stability"
  • Avoid: Generic descriptions without measurable outcomes
- Quantify improvements where possible (performance gains, bug reduction, code coverage, test reliability)
- Highlight architectural decisions and their technical rationale
- Be concise but comprehensive
- Use active voice and specific metrics when available
- Maximum 1000 words

SHOUT OUT GUIDELINES:
- Only highlight truly exceptional work (1-3 maximum, or none if not warranted)
- Distribute recognition across different contributors
- Focus on: complex problem-solving, code quality improvements, mentorship, critical fixes, or innovative solutions
- Format: "👏 @username - [specific achievement in 1-2 sentences]"

OUTPUT FORMAT:

## Summary of changes from {date_range}

**Development Focus:** [What were the main development themes and priorities this period?]

**Key Achievements:** [What significant features, improvements, or fixes were delivered?]

**Technical Health:** [What do the change patterns show about code quality and project evolution?]

**Notable PRs:** [Highlight 3-4 most impactful PRs based on complexity, user impact, or strategic importance -
include brief descriptions and PR links]

**Internal API Changes:** [List any API changes mentioned, with PR references]

**Shout Outs:** [List 0-3 shout outs, or "No specific shout outs this period" if none are warranted]

**Closing Thoughts:** [{selected_bonus_prompt}]

"""

        ai_response = self.ai_client.generate_with_retry(prompt, "phase2")

        if not ai_response:
            logging.error("AI generation failed for collection summary")
            return None

        return ai_response

    def _prepare_collection_context(self, pr_summaries: list[PRSummary], date_range: str, repository: str) -> str:
        """Prepare comprehensive context for collection summary."""

        stats = {"total_prs": len(pr_summaries), "by_category": {}, "by_impact": {}, "by_author": {}}

        for pr in pr_summaries:
            stats["by_category"][pr.category] = stats["by_category"].get(pr.category, 0) + 1
            stats["by_impact"][pr.impact_level] = stats["by_impact"].get(pr.impact_level, 0) + 1
            stats["by_author"][pr.author] = stats["by_author"].get(pr.author, 0) + 1

        top_authors = sorted(stats["by_author"].items(), key=lambda x: x[1], reverse=True)[:5]

        # Group PRs by category
        by_category = {}
        for pr in pr_summaries:
            if pr.category not in by_category:
                by_category[pr.category] = []
            by_category[pr.category].append(pr)

        context_lines = [
            f"**Repository:** {repository}",
            f"**Period:** {date_range}",
            f"**Total PRs:** {stats['total_prs']}",
            f"**Categories:** {dict(stats['by_category'])}",
            f"**Impact Levels:** {dict(stats['by_impact'])}",
            f"**Top Contributors:** {dict(top_authors)}",
            "",
            "**Detailed PR Context:**",
        ]

        for category, prs in by_category.items():
            context_lines.append(f"\n**{category.upper()} ({len(prs)} PRs):**")
            for pr in prs[:6]:  # Top 6 per category
                context_lines.append(f"• #{pr.pr_number}: {pr.title} ({pr.impact_level}) by {pr.author}")
                context_lines.append(f"  Summary: {pr.summary[:120]}...")
                if pr.developer_impact and pr.developer_impact != "Minimal developer impact.":
                    context_lines.append(f"  Impact: {pr.developer_impact[:80]}...")

        return "\n".join(context_lines)


def create_discord_summary(
    pr_summaries: list[PRSummary], collection_summary: str, date_range: str, github_run_url: str
) -> str:
    """Create Discord-formatted summary with enhanced formatting and shout outs."""

    stats = {"by_category": {}, "by_impact": {}}
    for pr in pr_summaries:
        stats["by_category"][pr.category] = stats["by_category"].get(pr.category, 0) + 1
        stats["by_impact"][pr.impact_level] = stats["by_impact"].get(pr.impact_level, 0) + 1

    lines = [
        f"📊 **Enhanced PR Summary Report** • {date_range}",
        "",
        "**📈 Statistics**",
        f"• Total PRs: {len(pr_summaries)}",
        f"• Categories: {', '.join(f'{k}: {v}' for k, v in stats['by_category'].items())}",
        f"• Impact: {', '.join(f'{k}: {v}' for k, v in stats['by_impact'].items())}",
        f"• Generated: <t:{int(time.time())}:R> using Gemini 2.5 with full context analysis",
        "",
        collection_summary,
        "",
        f"📦 [**Download Complete Analysis (pr-summary-N.zip)**]({github_run_url})",
    ]

    return "\n".join(lines)


def main():
    """Main entry point."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Get configuration
    api_key = os.getenv("GEMINI_API_KEY")
    pr_digest_file = os.getenv("PR_DIGEST_FILE", "pr_digest_output.json")
    date_range = os.getenv("DATE_RANGE", "")
    repository = os.getenv("REPOSITORY", "")
    github_run_url = (
        f"{os.getenv('GITHUB_SERVER_URL', '')}/{os.getenv('GITHUB_REPOSITORY', '')}"
        f"/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
    )

    if not api_key:
        print("Error: GEMINI_API_KEY not provided")
        sys.exit(1)

    if not Path(pr_digest_file).exists():
        print(f"Error: PR digest file {pr_digest_file} not found")
        sys.exit(1)

    # Load PR data
    with open(pr_digest_file, "r") as f:
        pr_data_list = json.load(f)

    if not pr_data_list:
        print("No PRs found in digest")
        sys.exit(0)

    # Initialize generator
    generator = PRSummaryGenerator(api_key)

    print(f"Processing {len(pr_data_list)} PRs with enhanced analysis...")

    # Process PRs with caching and concurrency
    pr_summaries = generator.process_prs_with_cache(pr_data_list)

    if not pr_summaries:
        print("No summaries generated")
        sys.exit(1)

    # Generate collection summary and shout outs
    print("Generating collection summary and shout outs...")
    collection_summary = generator.generate_collection_summary(pr_summaries, date_range, repository)

    # Create Discord-formatted output
    discord_summary = create_discord_summary(pr_summaries, collection_summary, date_range, github_run_url)

    # Save Discord summary
    with open("pr_summary_output.txt", "w") as f:
        f.write(discord_summary)

    # Save structured data (only one JSON file)
    with open("pr_summary_data.json", "w") as f:
        json.dump([asdict(pr) for pr in pr_summaries], f, indent=2)

    # GitHub outputs
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write("summary-file=pr_summary_output.txt\n")
            f.write("data-file=pr_summary_data.json\n")
            f.write(f"individual-summaries={len(pr_summaries)}\n")

    print(f"✅ Generated {len(pr_summaries)} comprehensive summaries with caching and concurrency")
    print(f"📁 Artifacts saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()
