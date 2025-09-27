#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-generativeai>=0.3.0",
# ]
# ///

import json
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from gemini_analyze_pr import PRAnalyzer, PRSummary, load_pr_summary, save_pr_summary
from gemini_client import GeminiAIClient


class PreviousReportExtractor:
    """Extracts and processes previous report artifacts (newsletters or author reports)."""

    def __init__(self, report_type: str = "newsletter", report_dir: str | None = None):
        self.report_type = report_type
        self.report_dir = Path(report_dir or f"previous-{report_type}s")

    def _extract_date_from_content(self, content: str) -> Optional[datetime]:
        """Extract end date from newsletter content as fallback."""

        # Look for date patterns in the newsletter header
        # Pattern 1: "• Month DD, YYYY to Month DD, YYYY"
        pattern1 = r"•\s*(\w+\s+\d{1,2},\s+\d{4})\s+to\s+(\w+\s+\d{1,2},\s+\d{4})"
        match = re.search(pattern1, content[:500])  # Check first 500 chars

        if match:
            try:
                end_date_str = match.group(2)
                end_date = datetime.strptime(end_date_str, "%B %d, %Y")
                return end_date
            except ValueError:
                logging.debug(f"Failed to parse date from content pattern: {match.group(2)}")

        # Pattern 2: Try ISO date format if present
        # "Historical: YYYY-MM-DD to YYYY-MM-DD"
        pattern2 = r"(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})"
        match = re.search(pattern2, content[:500])

        if match:
            try:
                end_date_str = match.group(2)
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                return end_date
            except ValueError:
                logging.debug(f"Failed to parse ISO date from content: {match.group(2)}")

        return None

    def extract_report_summaries(self) -> list[dict[str, str]]:
        """Extract summaries from previous report artifacts."""
        if not self.report_dir.exists():
            logging.info(f"Previous {self.report_type}s directory '{self.report_dir}' not found")
            return []

        summaries = []
        zip_files = list(self.report_dir.glob("*.zip"))

        if not zip_files:
            logging.info(f"No {self.report_type} artifacts found")
            return []

        logging.info(f"Found {len(zip_files)} {self.report_type} artifacts to process")

        for zip_path in zip_files:
            try:
                artifact_name = zip_path.stem

                # Extract date from artifact name if possible
                # Expected format: newsletter-YYYY-MM-DD-to-YYYY-MM-DD
                artifact_date = None
                if "newsletter-" in artifact_name and "-to-" in artifact_name:
                    try:
                        # Extract end date from artifact name
                        date_parts = artifact_name.split("-to-")
                        if len(date_parts) == 2:
                            end_date_str = date_parts[1]
                            artifact_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                    except ValueError:
                        logging.debug(f"Could not parse date from artifact name: {artifact_name}")

                run_id = artifact_name.split("_")[-1] if "_" in artifact_name else artifact_name

                with zipfile.ZipFile(zip_path, "r") as zf:
                    # Look for the appropriate summary file based on report type
                    if self.report_type == "newsletter":
                        summary_file = self._find_file_in_zip(zf, "discord_summary_output.txt")
                    elif self.report_type == "author-report":
                        summary_file = self._find_file_in_zip(zf, "author_report_output.md")
                    else:
                        logging.warning(f"Unknown report type: {self.report_type}")
                        continue

                    if summary_file:
                        content = zf.read(summary_file).decode("utf-8")
                        file_info = zf.getinfo(summary_file)
                        file_date = file_info.date_time
                        timestamp = datetime(*file_date[:6])

                        # If we couldn't get date from filename, try to extract from content
                        if not artifact_date:
                            artifact_date = self._extract_date_from_content(content)
                            if artifact_date:
                                logging.debug(f"Extracted date from content for {artifact_name}: {artifact_date}")

                        summaries.append(
                            {
                                "content": content,
                                "run_id": run_id,
                                "date": timestamp.isoformat(),
                                "artifact_date": artifact_date.isoformat() if artifact_date else None,
                                "artifact_name": artifact_name,
                                "report_type": self.report_type,
                            }
                        )

                        logging.info(f"✅ Extracted {self.report_type} summary from {artifact_name}")
                    else:
                        logging.warning(f"⚠️ No summary file found in {artifact_name}")

            except Exception as e:
                logging.error(f"❌ Error processing {zip_path.name}: {e}")
                continue

        summaries.sort(key=lambda x: x["artifact_date"] or x["date"])
        logging.info(f"Successfully extracted {len(summaries)} {self.report_type} summaries")
        return summaries

    def _find_file_in_zip(self, zf: zipfile.ZipFile, target_filename: str) -> str | None:
        """Find a file in the zip archive that ends with the target filename."""
        for name in zf.namelist():
            if name.endswith(target_filename):
                return name
        return None

    def get_recent_summaries(
        self, max_summaries: int = 3, author: str | None = None, end_date: Optional[datetime] = None
    ) -> list[dict[str, str]]:
        """Get formatted context from previous summaries for AI prompt.

        Args:
            max_summaries: Maximum number of summaries to return
            author: Filter by author (for author reports)
            end_date: Only include summaries before this date (for historical runs)
        """
        summaries = self.extract_report_summaries()

        if not summaries:
            return []

        # Filter by author if specified (for author reports)
        if author and self.report_type == "author-report":
            summaries = [s for s in summaries if author.lower() in s["artifact_name"].lower()]

        # Filter by date if specified (for historical runs)
        if end_date:
            filtered_summaries = []
            for summary in summaries:
                # Use artifact_date if available, otherwise fall back to file date
                date_str = summary.get("artifact_date") or summary["date"]
                summary_date = datetime.fromisoformat(date_str)

                # Only include summaries from before the end date
                if summary_date < end_date:
                    filtered_summaries.append(summary)
                else:
                    logging.debug(f"Excluding future summary from {date_str} (after {end_date})")

            summaries = filtered_summaries
            logging.info(f"After date filtering: {len(summaries)} summaries remain")

        return summaries[-max_summaries:] if len(summaries) > max_summaries else summaries


class PRDigestAnalyzer:
    """Main orchestrator for analyzing PR digests."""

    def __init__(self, api_key: str, summaries_dir: Path = Path("pr-summaries")):
        self.ai_client = GeminiAIClient(api_key)
        self.pr_analyzer = PRAnalyzer(self.ai_client)
        self.summaries_dir = summaries_dir

    def load_summaries(self, pr_numbers: list[int]) -> list[PRSummary]:
        """Load multiple PR summaries from cache."""
        summaries = []
        for pr_num in pr_numbers:
            summary = load_pr_summary(pr_num, self.summaries_dir)
            if summary:
                summaries.append(summary)
                logging.info(f"Loaded cached summary for PR #{pr_num}")
            else:
                logging.warning(f"Failed to load cached summary for PR #{pr_num}")
        return summaries

    def analyze_digest(
        self, new_prs: list[dict], cached_pr_numbers: list[int], use_parallel: bool = True, max_workers: int = 5
    ) -> list[PRSummary]:
        """Analyze new PRs and combine with cached summaries."""
        all_summaries = []

        # Step 1: Load cached summaries
        if cached_pr_numbers:
            logging.info(f"Loading {len(cached_pr_numbers)} cached PR summaries...")
            cached_summaries = self.load_summaries(cached_pr_numbers)
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

            # Convert to PRSummary and add to results
            for summary_dict in new_summaries:
                if summary_dict:
                    summary_data = PRSummary(
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
                        # Quality metrics
                        code_quality_score=summary_dict.get("code_quality_score"),
                        code_quality_notes=summary_dict.get("code_quality_notes"),
                        best_practices_score=summary_dict.get("best_practices_score"),
                        best_practices_notes=summary_dict.get("best_practices_notes"),
                        documentation_score=summary_dict.get("documentation_score"),
                        documentation_notes=summary_dict.get("documentation_notes"),
                        testing_score=summary_dict.get("testing_score"),
                        testing_notes=summary_dict.get("testing_notes"),
                        strengths=summary_dict.get("strengths"),
                        areas_for_improvement=summary_dict.get("areas_for_improvement"),
                        additions=summary_dict.get("additions"),
                        deletions=summary_dict.get("deletions"),
                        files_changed=summary_dict.get("files_changed"),
                        test_coverage_impact=summary_dict.get("test_coverage_impact"),
                    )
                    all_summaries.append(summary_data)

        # Sort by PR number (newest first)
        all_summaries.sort(key=lambda x: x.pr_number, reverse=True)

        logging.info(
            f"Total summaries: {len(all_summaries)} ({len([s for s in all_summaries if s.source == 'cache'])} cached,"
            f" {len([s for s in all_summaries if s.source == 'new'])} new)"
        )

        return all_summaries

    def _process_parallel(
        self,
        prs: list[dict],
        max_workers: int,
    ) -> list[dict]:
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

    def _process_sequential(self, prs: list[dict]) -> list[dict]:
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


def load_digest_data(pr_digest_file: str, stats_file: str) -> tuple[list[dict], dict]:
    """Load PR digest and statistics files."""
    if not Path(pr_digest_file).exists():
        raise FileNotFoundError(f"PR digest file {pr_digest_file} not found")

    if not Path(stats_file).exists():
        raise FileNotFoundError(f"Stats file {stats_file} not found")

    with open(pr_digest_file, "r") as f:
        new_prs = json.load(f)

    with open(stats_file, "r") as f:
        stats = json.load(f)

    return new_prs, stats


def save_new_summaries(summaries: list[PRSummary], summaries_dir: Path = Path("pr-summaries")) -> int:
    """Save newly processed PR summaries to individual files."""
    summaries_dir.mkdir(exist_ok=True)
    new_summaries = [s for s in summaries if s.source == "new"]

    if new_summaries:
        logging.info(f"Saving {len(new_summaries)} new PR summaries to {summaries_dir}/...")
        for pr_summary in new_summaries:
            filepath = save_pr_summary(pr_summary, summaries_dir)
            logging.debug(f"Saved {filepath}")

    return len(new_summaries)


def save_structured_data(summaries: list[PRSummary], output_file: str = "pr_summary_data.json"):
    """Save all summaries as structured JSON data."""
    with open(output_file, "w") as f:
        json.dump([asdict(pr) for pr in summaries], f, indent=2)
    logging.info(f"Saved {output_file}")


def create_discord_summary(
    pr_summaries: list[PRSummary],
    newsletter_content: str,
    date_range: str,
    github_run_url: str,
    stats: dict[str, int],
    github_repository: str,
) -> str:
    """Create Discord-formatted summary with statistics."""
    category_stats = {}
    impact_stats = {}
    for pr in pr_summaries:
        category_stats[pr.category] = category_stats.get(pr.category, 0) + 1
        impact_stats[pr.impact_level] = impact_stats.get(pr.impact_level, 0) + 1

    lines = [
        f"📊 **{github_repository} Newsletter** • {date_range}",
        "",
        "**📈 Statistics**",
        f"• Total PRs analyzed: {len(pr_summaries)}",
        f"• New PRs summarized: {stats.get('new_prs_to_fetch', 0)}",
        f"• Previously cached: {stats.get('cached_pr_count', 0)}",
        f"• Categories: {', '.join(f'{k}: {v}' for k, v in category_stats.items())}",
        f"• Impact: {', '.join(f'{k}: {v}' for k, v in impact_stats.items())}",
        f"• Generated: <t:{int(time.time())}:R> using Gemini with full context analysis",
        "",
        newsletter_content,
        "",
        f"📦 [**Download Complete Analysis**]({github_run_url})",
    ]

    return "\n".join(lines)
