#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-generativeai>=0.3.0",
# ]
# ///

import json
import logging
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from gemini_analyze_pr import PRAnalyzer, PRSummary
from gemini_client import MODEL_CONFIG, GeminiAIClient


@dataclass
class PRSummaryData:
    """Extended PRSummary with source tracking."""

    pr_number: int
    title: str
    summary: str
    key_changes: List[str]
    developer_impact: str
    technical_notes: Optional[str]
    category: str
    impact_level: str
    author: str
    merged_at: str
    html_url: str
    source: str  # "cache" or "new"

    def to_pr_summary(self) -> PRSummary:
        """Convert to PRSummary object."""
        return PRSummary(
            pr_number=self.pr_number,
            title=self.title,
            summary=self.summary,
            key_changes=self.key_changes,
            developer_impact=self.developer_impact,
            technical_notes=self.technical_notes,
            category=self.category,
            impact_level=self.impact_level,
            author=self.author,
            merged_at=self.merged_at,
            html_url=self.html_url,
        )


class CachedSummaryLoader:
    """Loads PR summaries from cache files."""

    def __init__(self, summaries_dir: Path = Path("pr-summaries")):
        self.summaries_dir = summaries_dir

    def load_summary(self, pr_number: int) -> Optional[PRSummaryData]:
        """Load a single PR summary from cache."""
        summary_file = self.summaries_dir / f"pr_{pr_number}.txt"

        if not summary_file.exists():
            return None

        try:
            content = summary_file.read_text(encoding="utf-8")

            # Parse the structured summary file
            sections = {
                "title": "",
                "author": "",
                "merged_at": "",
                "category": "",
                "impact_level": "",
                "html_url": "",
                "summary": "",
                "key_changes": [],
                "developer_impact": "",
                "technical_notes": "",
            }

            current_section = None
            lines = content.split("\n")

            for line in lines:
                # Parse header
                if line.startswith(f"PR #{pr_number}: "):
                    sections["title"] = line.split(": ", 1)[1]
                elif line.startswith("Author: "):
                    sections["author"] = line.replace("Author: ", "").strip()
                elif line.startswith("Merged: "):
                    sections["merged_at"] = line.replace("Merged: ", "").strip()
                elif line.startswith("Category: "):
                    sections["category"] = line.replace("Category: ", "").strip().lower()
                elif line.startswith("Impact: "):
                    sections["impact_level"] = line.replace("Impact: ", "").strip().lower()
                elif line.startswith("GitHub: "):
                    sections["html_url"] = line.replace("GitHub: ", "").strip()

                # Section markers
                elif line.strip() == "SUMMARY":
                    current_section = "summary"
                elif line.strip() == "KEY CHANGES":
                    current_section = "key_changes"
                elif line.strip() == "DEVELOPER IMPACT":
                    current_section = "developer_impact"
                elif line.strip() == "TECHNICAL NOTES":
                    current_section = "technical_notes"
                elif line.startswith("=" * 20):
                    current_section = None

                # Content within sections
                elif current_section and line.strip():
                    if current_section == "key_changes" and line.startswith("• "):
                        sections["key_changes"].append(line[2:].strip())
                    elif current_section in ["summary", "developer_impact", "technical_notes"]:
                        if sections[current_section]:
                            sections[current_section] += " "
                        sections[current_section] += line.strip()

            # Create PRSummaryData object
            return PRSummaryData(
                pr_number=pr_number,
                title=sections["title"],
                summary=sections["summary"],
                key_changes=sections["key_changes"],
                developer_impact=sections["developer_impact"],
                technical_notes=sections["technical_notes"] if sections["technical_notes"] else None,
                category=sections["category"],
                impact_level=sections["impact_level"],
                author=sections["author"],
                merged_at=sections["merged_at"],
                html_url=sections["html_url"],
                source="cache",
            )

        except Exception as e:
            logging.error(f"Error loading cached summary for PR #{pr_number}: {e}")
            return None

    def load_summaries(self, pr_numbers: List[int]) -> List[PRSummaryData]:
        """Load multiple PR summaries from cache."""
        summaries = []
        for pr_num in pr_numbers:
            summary = self.load_summary(pr_num)
            if summary:
                summaries.append(summary)
                logging.info(f"Loaded cached summary for PR #{pr_num}")
            else:
                logging.warning(f"Failed to load cached summary for PR #{pr_num}")
        return summaries


class PreviousNewsletterExtractor:
    """Extracts and processes previous newsletter artifacts."""

    def __init__(self, newsletter_dir: str = "previous-newsletters"):
        self.newsletter_dir = Path(newsletter_dir)

    def extract_discord_summaries(self) -> List[Dict[str, str]]:
        """Extract discord summaries from previous newsletter artifacts."""
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
                artifact_name = zip_path.stem
                run_id = artifact_name.split("_")[-1] if "_" in artifact_name else artifact_name

                with zipfile.ZipFile(zip_path, "r") as zf:
                    discord_file = None
                    for name in zf.namelist():
                        if name.endswith("discord_summary_output.txt"):
                            discord_file = name
                            break

                    if discord_file:
                        content = zf.read(discord_file).decode("utf-8")
                        file_info = zf.getinfo(discord_file)
                        file_date = file_info.date_time
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

        summaries.sort(key=lambda x: x["date"])
        logging.info(f"Successfully extracted {len(summaries)} discord summaries")
        return summaries

    def get_previous_summaries_context(self, max_summaries: int = 3) -> str:
        """Get formatted context from previous summaries for AI prompt."""
        summaries = self.extract_discord_summaries()

        if not summaries:
            return ""

        recent_summaries = summaries[-max_summaries:] if len(summaries) > max_summaries else summaries

        all_shout_outs = []
        for summary in recent_summaries:
            content = summary["content"]
            if "**Shout Outs:**" in content:
                shout_section = content.split("**Shout Outs:**")[1].split("**")[0]
                import re

                mentioned_users = re.findall(r"👏 @(\w+)", shout_section)
                all_shout_outs.extend(mentioned_users)

        context_parts = [
            "\n**PREVIOUS NEWSLETTER SUMMARIES:**",
            f"(Showing {len(recent_summaries)} most recent newsletters for context and continuity)",
            "",
        ]

        if all_shout_outs:
            context_parts.extend(
                [
                    "**Recent Shout Outs Given:**",
                    f"These developers were recognized in recent newsletters: {', '.join(set(all_shout_outs))}",
                    "(Consider recognizing different contributors to distribute appreciation across the team)",
                    "",
                ]
            )

        for summary in recent_summaries:
            context_parts.extend(
                [
                    f"---Newsletter from {summary['date']}---",
                    summary["content"][:1500],
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
        self.newsletter_extractor = PreviousNewsletterExtractor()

    def prepare_context(self, pr_summaries: List[PRSummary], date_range: str, repository: str) -> str:
        """Prepare context for collection analysis."""
        stats = {"total_prs": len(pr_summaries), "by_category": {}, "by_impact": {}, "by_author": {}}

        for pr in pr_summaries:
            stats["by_category"][pr.category] = stats["by_category"].get(pr.category, 0) + 1
            stats["by_impact"][pr.impact_level] = stats["by_impact"].get(pr.impact_level, 0) + 1
            stats["by_author"][pr.author] = stats["by_author"].get(pr.author, 0) + 1

        top_authors = sorted(stats["by_author"].items(), key=lambda x: x[1], reverse=True)[:5]

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
            for pr in prs[:6]:
                context_lines.append(f"• #{pr.pr_number}: {pr.title} ({pr.impact_level}) by {pr.author}")
                context_lines.append(f"  Summary: {pr.summary[:120]}...")
                if pr.developer_impact and pr.developer_impact != "Minimal developer impact.":
                    context_lines.append(f"  Impact: {pr.developer_impact[:80]}...")

        return "\n".join(context_lines)

    def generate_collection_summary(self, pr_summaries: List[PRSummary], date_range: str, repository: str) -> str:
        """Generate a comprehensive collection summary."""
        context = self.prepare_context(pr_summaries, date_range, repository)
        previous_context = self.newsletter_extractor.get_previous_summaries_context()

        bonus_prompts = [
            "A haiku capturing the essence of this development cycle - focus on the rhythm of progress, "
            "bugs conquered, or features born",
            "A team affirmation based on the collective accomplishments and growth demonstrated in this period's work",
            "A zen koan that emerges from the deepest technical challenge or breakthrough of this period - make it "
            "thought-provoking for engineers",
            "A brief reflection on the engineering patterns and evolution visible in this collection of changes",
        ]

        selected_bonus_prompt = random.choice(bonus_prompts)

        prompt = f"""
You are creating an executive summary of development activity for {repository} from {date_range}.

{previous_context}

INPUT DATA: Below you'll find PR summaries with titles, descriptions, authors, file changes, and technical details:

{context}

AUDIENCE & TONE: Technical and direct - written for engineering and research staff who understand code architecture,
implementation details, and technical tradeoffs.

CONTEXT AWARENESS & NARRATIVE CONTINUITY:
- Tell the ongoing story of our development process as it evolves - each newsletter should feel like a
  chapter in a larger narrative
- Reference significant trends or changes compared to previous newsletters to show progression
- Highlight any continuing work or resolved issues from previous periods
- Connect current achievements to past challenges or initiatives when relevant
- Maintain consistent tone and perspective across newsletters while letting the story naturally evolve
- Show how the team and codebase are growing and adapting over time

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
- IMPORTANT: Review previous newsletters and avoid repeatedly recognizing the same contributors
- If a developer received a shout out in recent newsletters, prioritize recognizing others who have done notable work
- Aim to distribute recognition across the entire team over time
- Focus on: complex problem-solving, code quality improvements, mentorship, critical fixes, or innovative solutions
- Format: "👏 @username - [specific achievement in 1-2 sentences]"

OUTPUT FORMAT:

## Summary of changes from {date_range}

**Development Focus:** [In 2-3 concise sentences, describe what the team actually built or fixed this period. Be specific and direct - avoid abstract themes or philosophical descriptions. Lead with concrete work: "We shipped X, refactored Y, and fixed Z" rather than "This period was characterized by..." Vary your language across summaries.]

**Key Achievements:** [What significant features, improvements, or fixes were delivered?]

**Technical Health:** [What do the change patterns show about code quality and project evolution?]

**Notable PRs:** [Highlight 3-4 most impactful PRs based on complexity, user impact, or strategic importance -
include brief descriptions and PR links]

**Internal API Changes:** [List any API changes mentioned, with PR references]

**Shout Outs:** [List 0-3 shout outs, or "No specific shout outs this period" if none are warranted]

**Closing Thoughts:** [{selected_bonus_prompt}]
"""

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
        pr_summaries: List[PRSummary],
        collection_summary: str,
        date_range: str,
        github_run_url: str,
        stats: Dict[str, int],
        github_repository: str,
    ) -> str:
        """Create Discord-formatted summary with statistics."""
        category_stats = {}
        impact_stats = {}
        for pr in pr_summaries:
            category_stats[pr.category] = category_stats.get(pr.category, 0) + 1
            impact_stats[pr.impact_level] = impact_stats.get(pr.impact_level, 0) + 1

        lines = [
            f"📊 ** {github_repository} Newsletter ** • {date_range}",
            "",
            "**📈 Statistics**",
            f"• Total PRs analyzed: {len(pr_summaries)}",
            f"• New PRs summarized: {stats.get('new_prs_to_fetch', 0)}",
            f"• Previously cached: {stats.get('cached_pr_count', 0)}",
            f"• Categories: {', '.join(f'{k}: {v}' for k, v in category_stats.items())}",
            f"• Impact: {', '.join(f'{k}: {v}' for k, v in impact_stats.items())}",
            f"• Generated: <t:{int(time.time())}:R> using Gemini 2.5 with full context analysis",
            "",
            collection_summary,
            "",
            f"📦 [**Download Complete Analysis**]({github_run_url})",
        ]

        return "\n".join(lines)


class PRDigestAnalyzer:
    """Main orchestrator for analyzing PR digests."""

    def __init__(self, api_key: str):
        self.ai_client = GeminiAIClient(api_key)
        self.pr_analyzer = PRAnalyzer(self.ai_client)
        self.collection_analyzer = CollectionAnalyzer(self.ai_client)
        self.output_formatter = OutputFormatter()
        self.cached_loader = CachedSummaryLoader()

    def analyze_digest(
        self, new_prs: List[Dict], cached_pr_numbers: List[int], use_parallel: bool = True, max_workers: int = 5
    ) -> List[PRSummaryData]:
        """Analyze new PRs and combine with cached summaries."""
        all_summaries = []

        # Step 1: Load cached summaries
        if cached_pr_numbers:
            logging.info(f"Loading {len(cached_pr_numbers)} cached PR summaries...")
            cached_summaries = self.cached_loader.load_summaries(cached_pr_numbers)
            all_summaries.extend(cached_summaries)
            logging.info(f"Successfully loaded {len(cached_summaries)} cached summaries")

        # Step 2: Process new PRs
        if new_prs:
            logging.info(f"Processing {len(new_prs)} new PRs...")

            if use_parallel and len(new_prs) > 2:
                logging.info(f"Using parallel processing with {max_workers} workers")
                new_summaries = self._process_parallel(new_prs, max_workers)
            else:
                logging.info("Using sequential processing")
                new_summaries = self._process_sequential(new_prs)

            # Convert to PRSummaryData and add to results
            for summary_dict in new_summaries:
                if summary_dict:
                    summary_data = PRSummaryData(
                        pr_number=summary_dict["pr_number"],
                        title=summary_dict["title"],
                        summary=summary_dict["summary"],
                        key_changes=summary_dict["key_changes"],
                        developer_impact=summary_dict["developer_impact"],
                        technical_notes=summary_dict.get("technical_notes"),
                        category=summary_dict["category"],
                        impact_level=summary_dict["impact_level"],
                        author=summary_dict["author"],
                        merged_at=summary_dict["merged_at"],
                        html_url=summary_dict["html_url"],
                        source="new",
                    )
                    all_summaries.append(summary_data)

        # Sort by PR number (newest first)
        all_summaries.sort(key=lambda x: x.pr_number, reverse=True)

        logging.info(
            f"Total summaries: {len(all_summaries)} ({len([s for s in all_summaries if s.source == 'cache'])} cached,"
            f" {len([s for s in all_summaries if s.source == 'new'])} new)"
        )

        return all_summaries

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
        "PR_DIGEST_STATS_FILE": "pr_digest_stats.json",
        "REPORT_PERIOD": "(unknown)",
    }

    # Parse configuration
    env_values = parse_config(required_vars, optional_vars)

    # Extract values
    api_key = env_values["GEMINI_API_KEY"]
    github_repository = env_values["GITHUB_REPOSITORY"]
    github_server_url = env_values["GITHUB_SERVER_URL"]
    github_run_id = env_values["GITHUB_RUN_ID"]
    pr_digest_file = env_values["PR_DIGEST_FILE"]
    stats_file = env_values["PR_DIGEST_STATS_FILE"]
    report_period = env_values["REPORT_PERIOD"]

    # Construct GitHub run URL
    github_run_url = f"{github_server_url}/{github_repository}/actions/runs/{github_run_id}"

    # Load PR digest and stats
    if not Path(pr_digest_file).exists():
        print(f"Error: PR digest file {pr_digest_file} not found")
        sys.exit(1)

    if not Path(stats_file).exists():
        print(f"Error: Stats file {stats_file} not found")
        sys.exit(1)

    with open(pr_digest_file, "r") as f:
        new_prs = json.load(f)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Check if there are any PRs in the time period
    if stats["total_prs_in_range"] == 0:
        print("No PRs found in the specified time period")

        # Create minimal output files
        with open("discord_summary_output.txt", "w") as f:
            f.write(f"📊 **PR Summary Report** • {report_period}\n\n")
            f.write("ℹ️ No PRs were merged during this period.\n")

        with open("collection_summary_output.txt", "w") as f:
            f.write("No PRs to summarize in this period.")

        with open("pr_summary_data.json", "w") as f:
            json.dump([], f)

        print("Created minimal output files")
        sys.exit(0)

    cached_pr_numbers = stats.get("cached_pr_numbers", [])

    print("Processing PR digest:")
    print(f"  - Total PRs in period: {stats['total_prs_in_range']}")
    print(f"  - New PRs to analyze: {len(new_prs)}")
    print(f"  - Cached PRs to load: {len(cached_pr_numbers)}")

    # Initialize analyzer and process
    analyzer = PRDigestAnalyzer(api_key)
    all_summaries = analyzer.analyze_digest(new_prs, cached_pr_numbers)

    if not all_summaries:
        print("No summaries generated")
        sys.exit(1)

    print(f"✅ Processed {len(all_summaries)} total PR summaries")

    # Convert to PRSummary objects for collection analysis
    pr_summaries = [s.to_pr_summary() for s in all_summaries]

    # Save individual PR files for newly processed PRs only
    pr_summaries_dir = Path("pr-summaries")
    pr_summaries_dir.mkdir(exist_ok=True)

    new_summaries = [s for s in all_summaries if s.source == "new"]
    if new_summaries:
        print(f"Saving {len(new_summaries)} new PR summaries to {pr_summaries_dir}/...")
        for summary_data in new_summaries:
            pr_summary = summary_data.to_pr_summary()
            filepath = analyzer.output_formatter.save_individual_pr_file(pr_summary, pr_summaries_dir)
            logging.debug(f"Saved {filepath}")
        print(f"✅ Saved {len(new_summaries)} new PR files")

    # Save structured data (all summaries)
    with open("pr_summary_data.json", "w") as f:
        json.dump([asdict(pr.to_pr_summary()) for pr in all_summaries], f, indent=2)
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
        pr_summaries, collection_summary, report_period, github_run_url, stats, github_repository
    )

    with open("discord_summary_output.txt", "w") as f:
        f.write(discord_summary)
    print("✅ Saved discord_summary_output.txt")

    print("🎉 PR digest analysis complete!")


if __name__ == "__main__":
    main()
