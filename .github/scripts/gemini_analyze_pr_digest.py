#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-generativeai>=0.3.0",
# ]
# ///

import json
import logging
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from gemini_analyze_pr import PRAnalyzer, PRSummary
from gemini_client import MODEL_CONFIG, GeminiAIClient


class CacheManager:
    """Manages caching of PR summaries to avoid re-analyzing unchanged PRs."""

    def __init__(self, cache_dir: str = ".pr-digest-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "pr_summaries_cache.json"

    def load_cache(self) -> Dict:
        """Load cached PR summaries."""
        force_refresh = os.getenv("FORCE_REFRESH", "false").lower() == "true"
        if force_refresh:
            logging.info("Force refresh enabled, skipping cache load")
            return {}

        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    cache_data = json.load(f)
                    if isinstance(cache_data, dict) and "model" in cache_data:
                        logging.info(f"Loaded cache with {len(cache_data.get('summaries', {}))} entries")
                        return cache_data
                return {}
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
        return {}

    def save_cache(self, summaries: Dict, model_name: str) -> None:
        """Save PR summaries to cache."""
        self.cache_dir.mkdir(exist_ok=True)

        cache_data = {"model": model_name, "generated_at": datetime.now().isoformat(), "summaries": summaries}

        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logging.info(f"Saved cache with {len(summaries)} entries")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def get_cache_key(self, pr_data: Dict) -> str:
        """Generate a cache key for a PR."""
        return f"{pr_data['number']}-{pr_data['merged_at']}"


class PreviousNewsletterExtractor:
    """Extracts and processes previous newsletter artifacts."""

    def __init__(self, newsletter_dir: str = "previous-newsletters"):
        self.newsletter_dir = Path(newsletter_dir)

    def extract_discord_summaries(self) -> List[Dict[str, str]]:
        """Extract discord summaries from previous newsletter artifacts.

        Returns:
            List of dicts with 'content', 'run_id', and 'date' keys, sorted by date (oldest first)
        """
        if not self.newsletter_dir.exists():
            logging.info(f"Previous newsletters directory '{self.newsletter_dir}' not found")
            return []

        summaries = []
        zip_files = list(self.newsletter_dir.glob("*.zip"))

        if not zip_files:
            logging.info("No newsletter artifacts found")
            return []

        logging.info(f"Found {len(zip_files)} newsletter artifacts to process")

        for zip_path in zip_files:
            try:
                # Extract run ID from filename (format: newsletter-X_RUNID.zip)
                artifact_name = zip_path.stem  # Remove .zip extension
                run_id = artifact_name.split("_")[-1] if "_" in artifact_name else artifact_name

                # Extract and read discord summary
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # Look for discord_summary_output.txt
                    discord_file = None
                    for name in zf.namelist():
                        if name.endswith("discord_summary_output.txt"):
                            discord_file = name
                            break

                    if discord_file:
                        # Read the content
                        content = zf.read(discord_file).decode("utf-8")

                        # Get file info for timestamp
                        file_info = zf.getinfo(discord_file)
                        file_date = file_info.date_time

                        # Convert to readable date string
                        from datetime import datetime

                        timestamp = datetime(*file_date[:6])

                        summaries.append(
                            {
                                "content": content,
                                "run_id": run_id,
                                "date": timestamp.isoformat(),
                                "artifact_name": artifact_name,
                            }
                        )

                        logging.info(f"✅ Extracted discord summary from {artifact_name}")
                    else:
                        logging.warning(f"⚠️ No discord_summary_output.txt found in {artifact_name}")

            except Exception as e:
                logging.error(f"❌ Error processing {zip_path.name}: {e}")
                continue

        # Sort by date (oldest first)
        summaries.sort(key=lambda x: x["date"])

        logging.info(f"Successfully extracted {len(summaries)} discord summaries")
        return summaries

    def get_previous_summaries_context(self, max_summaries: int = 3) -> str:
        """Get formatted context from previous summaries for AI prompt.

        Args:
            max_summaries: Maximum number of previous summaries to include

        Returns:
            Formatted string with previous summaries context
        """
        summaries = self.extract_discord_summaries()

        if not summaries:
            return ""

        # Take the most recent summaries (last N items since list is sorted oldest first)
        recent_summaries = summaries[-max_summaries:] if len(summaries) > max_summaries else summaries

        context_parts = [
            "\n**PREVIOUS NEWSLETTER SUMMARIES:**",
            f"(Showing {len(recent_summaries)} most recent newsletters for context and continuity)",
            "",
        ]

        for summary in recent_summaries:
            context_parts.extend(
                [
                    f"---Newsletter from {summary['date']}---",
                    summary["content"][:1500],  # Limit each summary to avoid prompt bloat
                    "...[truncated]" if len(summary["content"]) > 1500 else "",
                    "",
                ]
            )

        context_parts.append("---END OF PREVIOUS NEWSLETTERS---\n")

        return "\n".join(context_parts)


class CollectionAnalyzer:
    """Generates collection-level summaries and insights from multiple PR summaries."""

    def __init__(self, ai_client: GeminiAIClient):
        self.ai_client = ai_client

    def prepare_context(self, pr_summaries: List[PRSummary], date_range: str, repository: str) -> str:
        """Prepare context for collection analysis."""
        stats = {"total_prs": len(pr_summaries), "by_category": {}, "by_impact": {}, "by_author": {}}

        for pr in pr_summaries:
            stats["by_category"][pr.category] = stats["by_category"].get(pr.category, 0) + 1
            stats["by_impact"][pr.impact_level] = stats["by_impact"].get(pr.impact_level, 0) + 1
            stats["by_author"][pr.author] = stats["by_author"].get(pr.author, 0) + 1

        top_authors = sorted(stats["by_author"].items(), key=lambda x: x[1], reverse=True)[:5]

        # Group PRs by category for detailed context
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

        # Add detailed info for each category
        for category, prs in by_category.items():
            context_lines.append(f"\n**{category.upper()} ({len(prs)} PRs):**")
            for pr in prs[:6]:  # Limit to top 6 per category
                context_lines.append(f"• #{pr.pr_number}: {pr.title} ({pr.impact_level}) by {pr.author}")
                context_lines.append(f"  Summary: {pr.summary[:120]}...")
                if pr.developer_impact and pr.developer_impact != "Minimal developer impact.":
                    context_lines.append(f"  Impact: {pr.developer_impact[:80]}...")

        return "\n".join(context_lines)

    def generate_collection_summary(self, pr_summaries: List[PRSummary], date_range: str, repository: str) -> str:
        """Generate a comprehensive collection summary."""
        context = self.prepare_context(pr_summaries, date_range, repository)

        # Random creative element for closing thoughts
        bonus_prompts = [
            "A haiku capturing the essence of this development cycle - focus on the rhythm of progress, "
            "bugs conquered, or features born",
            "A team affirmation based on the collective accomplishments and growth demonstrated in this period's work",
            "A zen koan that emerges from the deepest technical challenge or breakthrough of this period - make it "
            "thought-provoking for engineers",
            "A brief reflection on the engineering patterns and evolution visible in this collection of changes",
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

        # Use best model for comprehensive collection analysis
        ai_response = self.ai_client.generate_with_retry(prompt, "best")

        if not ai_response:
            logging.error("AI generation failed for collection summary")
            return "Failed to generate collection summary"

        return ai_response


class OutputFormatter:
    """Handles various output formats for PR digest analysis."""

    @staticmethod
    def save_individual_pr_file(pr_summary: PRSummary, output_dir: Path) -> str:
        """Save individual PR summary as a text file."""
        filename = f"pr_{pr_summary.pr_number}.txt"
        filepath = output_dir / filename

        content = f"""PR #{pr_summary.pr_number}: {pr_summary.title}

Author: {pr_summary.author}
Merged: {pr_summary.merged_at}
Category: {pr_summary.category.title()}
Impact: {pr_summary.impact_level.title()}
GitHub: {pr_summary.html_url}

================================================================================

SUMMARY
{pr_summary.summary}

KEY CHANGES
{chr(10).join(f"• {change}" for change in pr_summary.key_changes)}

DEVELOPER IMPACT
{pr_summary.developer_impact}

TECHNICAL NOTES
{pr_summary.technical_notes if pr_summary.technical_notes else "No additional technical notes"}

================================================================================
Generated using {MODEL_CONFIG["default"]} on {datetime.now().isoformat()}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)

    @staticmethod
    def create_discord_summary(
        pr_summaries: List[PRSummary], collection_summary: str, date_range: str, github_run_url: str
    ) -> str:
        """Create Discord-formatted summary with statistics."""
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
            f"📦 [**Download Complete Analysis**]({github_run_url})",
        ]

        return "\n".join(lines)


class PRDigestAnalyzer:
    """Main orchestrator for analyzing PR digests with caching and parallel processing."""

    def __init__(self, api_key: str):
        self.ai_client = GeminiAIClient(api_key)
        self.pr_analyzer = PRAnalyzer(self.ai_client)
        self.cache_manager = CacheManager()
        self.collection_analyzer = CollectionAnalyzer(self.ai_client)
        self.output_formatter = OutputFormatter()

    def analyze_digest(
        self, pr_data_list: List[Dict], use_parallel: bool = True, max_workers: int = 5
    ) -> List[PRSummary]:
        """Analyze a digest of PRs with caching and optional parallel processing."""
        # Load cache
        cache_data = self.cache_manager.load_cache()
        model_name = MODEL_CONFIG["default"]

        if cache_data.get("model") != model_name:
            logging.info(f"Cache is for different model ({cache_data.get('model')}), regenerating")
            cached_summaries = {}
        else:
            cached_summaries = cache_data.get("summaries", {})

        updated_cache = cached_summaries.copy()
        results = []
        prs_to_process = []

        # Separate cached and uncached PRs
        for pr_data in pr_data_list:
            cache_key = self.cache_manager.get_cache_key(pr_data)
            if cache_key in cached_summaries:
                logging.info(f"Using cached summary for PR #{pr_data['number']}")
                results.append(PRSummary(**cached_summaries[cache_key]))
            else:
                prs_to_process.append(pr_data)

        logging.info(f"Found {len(results)} cached summaries, processing {len(prs_to_process)} new PRs")

        # Process uncached PRs
        if prs_to_process:
            if use_parallel and len(prs_to_process) > 2:
                logging.info(f"Using parallel processing with {max_workers} workers")
                new_summaries = self._process_parallel(prs_to_process, max_workers)
            else:
                logging.info("Using sequential processing")
                new_summaries = self._process_sequential(prs_to_process)

            # Add new summaries to results and cache
            for summary_data in new_summaries:
                if summary_data:
                    summary = PRSummary(**summary_data)
                    results.append(summary)

                    cache_key = self.cache_manager.get_cache_key(
                        {"number": summary.pr_number, "merged_at": summary.merged_at}
                    )
                    updated_cache[cache_key] = summary_data

            # Save updated cache
            self.cache_manager.save_cache(updated_cache, model_name)

        # Sort by PR number (newest first)
        results.sort(key=lambda x: x.pr_number, reverse=True)
        return results

    def _process_parallel(self, prs: List[Dict], max_workers: int) -> List[Dict]:
        """Process PRs in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pr = {executor.submit(self.pr_analyzer.analyze, pr): pr for pr in prs}

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

    def _process_sequential(self, prs: List[Dict]) -> List[Dict]:
        """Process PRs sequentially."""
        results = []

        for i, pr_data in enumerate(prs):
            logging.info(f"Processing PR {i + 1}/{len(prs)}: #{pr_data['number']}")
            try:
                result = self.pr_analyzer.analyze(pr_data)
                if result:
                    results.append(result)
                else:
                    logging.error(f"Failed to process PR #{pr_data['number']}")
            except Exception as e:
                logging.error(f"Error processing PR #{pr_data['number']}: {e}")

        return results


def main():
    """Main entry point for PR digest analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Import parse_config
    script_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    from utils.config import parse_config

    # Define required and optional environment variables
    required_vars = [
        "GEMINI_API_KEY",
        "GITHUB_REPOSITORY",
        "GITHUB_SERVER_URL",
        "GITHUB_RUN_ID",
    ]

    optional_vars = {
        "PR_DIGEST_FILE": "pr_digest_output.json",
        "REPORT_PERIOD": "(unknown)",
    }

    # Parse configuration
    env_values = parse_config(required_vars, optional_vars)

    # Extract values for easier use
    api_key = env_values["GEMINI_API_KEY"]
    github_repository = env_values["GITHUB_REPOSITORY"]
    github_server_url = env_values["GITHUB_SERVER_URL"]
    github_run_id = env_values["GITHUB_RUN_ID"]
    pr_digest_file = env_values["PR_DIGEST_FILE"]
    report_period = env_values["REPORT_PERIOD"]

    # Construct GitHub run URL
    github_run_url = f"{github_server_url}/{github_repository}/actions/runs/{github_run_id}"

    # Validate file existence
    if not Path(pr_digest_file).exists():
        print(f"Error: PR digest file {pr_digest_file} not found")
        sys.exit(1)

    # Load and validate PR data
    with open(pr_digest_file, "r") as f:
        pr_data_list = json.load(f)

    if not pr_data_list:
        print("No PRs found in digest")
        sys.exit(0)

    print(f"Processing {len(pr_data_list)} PRs with AI analysis...")

    # Initialize analyzer and process PRs
    analyzer = PRDigestAnalyzer(api_key)
    pr_summaries = analyzer.analyze_digest(pr_data_list)

    if not pr_summaries:
        print("No summaries generated")
        sys.exit(1)

    print(f"✅ Generated {len(pr_summaries)} PR summaries")

    # Create output directory and save individual PR files
    pr_summaries_dir = Path("pr-summaries")
    pr_summaries_dir.mkdir(exist_ok=True)

    print(f"Saving individual PR summaries to {pr_summaries_dir}/...")
    for pr_summary in pr_summaries:
        filepath = analyzer.output_formatter.save_individual_pr_file(pr_summary, pr_summaries_dir)
        logging.debug(f"Saved {filepath}")

    print(f"✅ Saved {len(pr_summaries)} individual PR files")

    # Save structured data
    with open("pr_summary_data.json", "w") as f:
        json.dump([asdict(pr) for pr in pr_summaries], f, indent=2)
    print("✅ Saved pr_summary_data.json")

    # Generate collection summary
    print("Generating collection summary...")
    collection_summary = analyzer.collection_analyzer.generate_collection_summary(
        pr_summaries, report_period, github_repository
    )

    with open("collection_summary_output.txt", "w") as f:
        f.write(collection_summary)
    print("✅ Saved collection_summary_output.txt")

    # Create Discord-formatted output
    print("Generating Discord summary...")
    discord_summary = analyzer.output_formatter.create_discord_summary(
        pr_summaries, collection_summary, report_period, github_run_url
    )

    with open("discord_summary_output.txt", "w") as f:
        f.write(discord_summary)
    print("✅ Saved discord_summary_output.txt")

    print("🎉 PR digest analysis complete!")


if __name__ == "__main__":
    main()
