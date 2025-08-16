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
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from gemini_analyze_pr import PRAnalyzer, PRSummary, load_pr_summary, save_pr_summary
from gemini_client import GeminiAIClient


class CachedSummaryLoader:
    """Loads PR summaries from cache files."""

    def __init__(self, summaries_dir: Path = Path("pr-summaries")):
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


class PreviousReportExtractor:
    """Extracts and processes previous report artifacts (newsletters or author reports)."""

    def __init__(self, report_type: str = "newsletter", report_dir: str | None = None):
        self.report_type = report_type
        self.report_dir = Path(report_dir or f"previous-{report_type}s")

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

                        summaries.append(
                            {
                                "content": content,
                                "run_id": run_id,
                                "date": timestamp.isoformat(),
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

        summaries.sort(key=lambda x: x["date"])
        logging.info(f"Successfully extracted {len(summaries)} {self.report_type} summaries")
        return summaries

    def _find_file_in_zip(self, zf: zipfile.ZipFile, target_filename: str) -> str | None:
        """Find a file in the zip archive that ends with the target filename."""
        for name in zf.namelist():
            if name.endswith(target_filename):
                return name
        return None

    def get_previous_summaries_context(self, max_summaries: int = 3, author: str | None = None) -> str:
        """Get formatted context from previous summaries for AI prompt."""
        summaries = self.extract_report_summaries()

        if not summaries:
            return ""

        # Filter by author if specified (for author reports)
        if author and self.report_type == "author-report":
            summaries = [s for s in summaries if author.lower() in s["artifact_name"].lower()]

        recent_summaries = summaries[-max_summaries:] if len(summaries) > max_summaries else summaries

        if self.report_type == "newsletter":
            return self._format_newsletter_context(recent_summaries)
        elif self.report_type == "author-report":
            return self._format_author_report_context(recent_summaries)
        else:
            return ""

    def _format_newsletter_context(self, summaries: list[dict]) -> str:
        """Format newsletter summaries for context."""
        all_shout_outs = []
        for summary in summaries:
            content = summary["content"]
            if "**Shout Outs:**" in content:
                shout_section = content.split("**Shout Outs:**")[1].split("**")[0]
                import re

                mentioned_users = re.findall(r"🎉 @(\w+)", shout_section)
                all_shout_outs.extend(mentioned_users)

        context_parts = [
            "\n**PREVIOUS NEWSLETTER SUMMARIES:**",
            f"(Showing {len(summaries)} most recent newsletters for context and continuity)",
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

        for summary in summaries:
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

    def _format_author_report_context(self, summaries: list[dict]) -> str:
        """Format author report summaries for context."""
        context_parts = ["**PREVIOUS PERFORMANCE REVIEWS:**"]

        for summary in summaries:
            # Extract key metrics from the markdown report
            content = summary["content"]
            period_match = re.search(r"\*\*Period:\*\*\s*(.+)", content)
            period = period_match.group(1) if period_match else "Unknown period"

            context_parts.extend(
                [
                    f"\n---Report from {summary['date']} (Period: {period})---",
                    summary["content"][:1000],
                    "...[truncated]" if len(summary["content"]) > 1000 else "",
                    "",
                ]
            )

        context_parts.append("**END OF PREVIOUS REVIEWS**\n")
        return "\n".join(context_parts)


class AuthorReportGenerator:
    """Generates comprehensive author-specific reports."""

    def __init__(self, ai_client: GeminiAIClient):
        self.ai_client = ai_client

    def calculate_author_stats(self, pr_summaries: list[PRSummary]) -> dict:
        """Calculate aggregate statistics for an author."""
        stats = {
            "total_prs": len(pr_summaries),
            "total_additions": 0,
            "total_deletions": 0,
            "total_files_changed": 0,
            "categories": {},
            "impact_levels": {},
            "avg_code_quality": 0,
            "avg_best_practices": 0,
            "avg_documentation": 0,
            "avg_testing": 0,
            "all_strengths": [],
            "all_improvements": [],
            "test_coverage_changes": {"increased": 0, "maintained": 0, "decreased": 0, "unknown": 0},
            # New statistics
            "avg_review_cycles": 0,
            "avg_pr_size": 0,  # Average lines changed per PR
            "consistency_score": 0,  # Based on quality score variance
            "quality_trend": "stable",  # improving, declining, stable
            "preferred_categories": [],  # Top 3 categories by frequency
            "impact_distribution": {  # Percentage of each impact level
                "minor": 0,
                "moderate": 0,
                "major": 0,
            },
            "code_to_test_ratio": 0,  # Ratio of test code to implementation code
            "documentation_rate": 0,  # Percentage of PRs with documentation updates
        }

        quality_scores = {
            "code_quality": [],
            "best_practices": [],
            "documentation": [],
            "testing": [],
        }

        review_cycles = []
        pr_sizes = []
        has_documentation_updates = 0
        test_file_changes = 0
        non_test_file_changes = 0

        for pr in pr_summaries:
            # Aggregate statistics
            if pr.additions:
                stats["total_additions"] += pr.additions
            if pr.deletions:
                stats["total_deletions"] += pr.deletions
            if pr.files_changed:
                stats["total_files_changed"] += pr.files_changed

            # Categories and impact
            stats["categories"][pr.category] = stats["categories"].get(pr.category, 0) + 1
            stats["impact_levels"][pr.impact_level] = stats["impact_levels"].get(pr.impact_level, 0) + 1

            # Quality scores
            if pr.code_quality_score:
                quality_scores["code_quality"].append(pr.code_quality_score)
            if pr.best_practices_score:
                quality_scores["best_practices"].append(pr.best_practices_score)
            if pr.documentation_score:
                quality_scores["documentation"].append(pr.documentation_score)
                if pr.documentation_score >= 7:  # Good documentation
                    has_documentation_updates += 1
            if pr.testing_score:
                quality_scores["testing"].append(pr.testing_score)

            # Strengths and improvements
            if pr.strengths:
                stats["all_strengths"].extend(pr.strengths)
            if pr.areas_for_improvement:
                stats["all_improvements"].extend(pr.areas_for_improvement)

            # Test coverage
            if pr.test_coverage_impact:
                stats["test_coverage_changes"][pr.test_coverage_impact] += 1

            # Review cycles
            if pr.review_cycles is not None:
                review_cycles.append(pr.review_cycles)

            # PR size
            if pr.additions is not None and pr.deletions is not None:
                pr_sizes.append(pr.additions + pr.deletions)

            # Test vs non-test changes (rough estimate based on category)
            if pr.category == "testing":
                test_file_changes += pr.files_changed or 0
            else:
                non_test_file_changes += pr.files_changed or 0
                # Check if non-test PRs included tests
                if pr.test_coverage_impact == "increased":
                    test_file_changes += 1  # Estimate

        # Calculate averages
        for metric, scores in quality_scores.items():
            if scores:
                stats[f"avg_{metric}"] = round(sum(scores) / len(scores), 1)

        # Calculate new metrics
        if review_cycles:
            stats["avg_review_cycles"] = round(sum(review_cycles) / len(review_cycles), 1)

        if pr_sizes:
            stats["avg_pr_size"] = round(sum(pr_sizes) / len(pr_sizes), 0)

        # Calculate consistency score (lower variance = higher consistency)
        if len(quality_scores["code_quality"]) > 1:
            # Calculate variance
            mean = stats["avg_code_quality"]
            variance = sum((score - mean) ** 2 for score in quality_scores["code_quality"]) / len(
                quality_scores["code_quality"]
            )
            # Convert to 0-10 scale (10 = most consistent)
            stats["consistency_score"] = round(max(0, 10 - variance), 1)

        # Determine quality trend
        if len(quality_scores["code_quality"]) >= 3:
            # Compare first third vs last third of PRs
            third = len(quality_scores["code_quality"]) // 3
            early_avg = sum(quality_scores["code_quality"][:third]) / third
            recent_avg = sum(quality_scores["code_quality"][-third:]) / len(quality_scores["code_quality"][-third:])

            if recent_avg > early_avg + 0.5:
                stats["quality_trend"] = "improving"
            elif recent_avg < early_avg - 0.5:
                stats["quality_trend"] = "declining"
            else:
                stats["quality_trend"] = "stable"

        # Preferred categories (top 3)
        sorted_categories = sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)
        stats["preferred_categories"] = [cat[0] for cat in sorted_categories[:3]]

        # Impact distribution
        total_prs = len(pr_summaries)
        if total_prs > 0:
            for impact in ["minor", "moderate", "major"]:
                count = stats["impact_levels"].get(impact, 0)
                stats["impact_distribution"][impact] = round((count / total_prs) * 100, 1)

        # Code to test ratio
        if non_test_file_changes > 0:
            stats["code_to_test_ratio"] = round(test_file_changes / non_test_file_changes, 2)

        # Documentation rate
        if total_prs > 0:
            stats["documentation_rate"] = round((has_documentation_updates / total_prs) * 100, 1)

        return stats

    def _get_previous_reports_context(self, author: str) -> str:
        """Extract context from previous author reports if available."""
        previous_reports_dir = Path(os.getenv("PREVIOUS_REPORTS_DIR", "previous-reports"))

        if not previous_reports_dir.exists():
            return ""

        context_parts = ["**PREVIOUS PERFORMANCE REVIEWS:**"]
        reports_found = False

        # Look for previous report artifacts
        for report_path in previous_reports_dir.glob("*/author_report_summary.json"):
            try:
                with open(report_path, "r") as f:
                    summary = json.load(f)

                if summary.get("author", "").lower() == author.lower():
                    reports_found = True
                    period = summary.get("period", "Unknown")
                    total_prs = summary.get("total_prs", 0)
                    avg_scores = summary.get("average_scores", {})
                    code_score = avg_scores.get("code_quality", "N/A")
                    test_score = avg_scores.get("testing", "N/A")
                    categories = summary.get("categories", {})
                    categories_str = ", ".join(f"{k}: {v}" for k, v in categories.items())

                    context_parts.extend(
                        [
                            f"\nPeriod: {period}",
                            f"- Total PRs: {total_prs}",
                            f"- Average Scores: Code {code_score}/10, Testing {test_score}/10",
                            f"- Categories: {categories_str}",
                        ]
                    )

            except Exception as e:
                logging.warning(f"Error reading previous report: {e}")
                continue

        if not reports_found:
            return ""

        context_parts.append("\n**END OF PREVIOUS REVIEWS**\n")
        return "\n".join(context_parts)

    def generate_author_report(self, pr_summaries: list[PRSummary], date_range: str, author: str) -> str:
        """Generate a comprehensive author report suitable for performance reviews."""
        # Handle empty PR list
        if not pr_summaries:
            return f"""# Performance Review: {author}
**Period:** {date_range}

## Summary
No pull requests were found for {author} during this period. This could indicate:
- The developer may have been working on other projects
- Contributing through pair programming or code reviews
- Focusing on non-code deliverables (documentation, planning, etc.)

## Recommendation
Schedule a discussion to understand the developer's contributions during this period.
"""

        # Handle single PR case with appropriate messaging
        if len(pr_summaries) == 1:
            pr = pr_summaries[0]
            return f"""# Performance Review: {author}
**Period:** {date_range}

## Summary
{author} contributed 1 pull request during this period.

### Pull Request Details
- **PR #{pr.pr_number}**: {pr.title}
- **Category**: {pr.category}
- **Impact**: {pr.impact_level}
- **Summary**: {pr.summary}

### Quality Metrics
- Code Quality: {pr.code_quality_score or "N/A"}/10
- Best Practices: {pr.best_practices_score or "N/A"}/10
- Documentation: {pr.documentation_score or "N/A"}/10
- Testing: {pr.testing_score or "N/A"}/10

### Notes
With only one PR in this period, a comprehensive performance review may require additional context from other sources
such as code reviews, team collaboration, and non-PR contributions.
"""

        # Validate author name matches PRs (defensive programming)
        pr_authors = {pr.author for pr in pr_summaries}
        if len(pr_authors) > 1 or (len(pr_authors) == 1 and author.lower() not in [a.lower() for a in pr_authors]):
            logging.warning(f"Author mismatch: requested '{author}' but PRs are by {pr_authors}")
            # Continue anyway but log the discrepancy

        # Calculate statistics with error handling
        try:
            stats = self.calculate_author_stats(pr_summaries)
        except Exception as e:
            logging.error(f"Error calculating statistics for {author}: {e}")
            # Provide basic stats as fallback
            stats = {
                "total_prs": len(pr_summaries),
                "total_additions": 0,
                "total_deletions": 0,
                "total_files_changed": 0,
                "categories": {},
                "impact_levels": {},
                "avg_code_quality": 0,
                "avg_best_practices": 0,
                "avg_documentation": 0,
                "avg_testing": 0,
                "all_strengths": [],
                "all_improvements": [],
                "test_coverage_changes": {"increased": 0, "maintained": 0, "decreased": 0, "unknown": 0},
            }

        # Prepare PR details for the prompt with error handling
        pr_details = []
        for _i, pr in enumerate(pr_summaries[:10]):  # Include top 10 PRs for context
            try:
                pr_detail = {
                    "number": pr.pr_number,
                    "title": pr.title,
                    "category": pr.category,
                    "impact": pr.impact_level,
                    "summary": pr.summary,
                    "key_changes": pr.key_changes,
                    "quality_scores": {
                        "code_quality": pr.code_quality_score,
                        "best_practices": pr.best_practices_score,
                        "documentation": pr.documentation_score,
                        "testing": pr.testing_score,
                    },
                    "strengths": pr.strengths or [],
                    "improvements": pr.areas_for_improvement or [],
                }
                pr_details.append(pr_detail)
            except AttributeError as e:
                logging.error(f"Error processing PR #{pr.pr_number}: Missing attribute {e}")
                continue

        # Ensure we have at least some PR details
        if not pr_details:
            logging.error(f"No valid PR details could be extracted for {author}")
            return f"""# Performance Review: {author}
**Period:** {date_range}

## Error
Unable to process PR details for performance review. The PR data may be corrupted or incomplete.

Total PRs found: {len(pr_summaries)}
Please check the data integrity and try again.
"""
        previous_reports_context = self._get_previous_reports_context(author)

        # Build the prompt with defensive checks
        prompt = f"""
You are creating a comprehensive performance review report for {author} covering {date_range}.

**STATISTICS:**
- Total PRs: {stats["total_prs"]}
- Code Changes: {stats["total_additions"]} additions, {stats["total_deletions"]} deletions across {
            stats["total_files_changed"]
        } files
- Work Distribution: {dict(stats["categories"]) if stats["categories"] else "No category data"}
- Impact Levels: {dict(stats["impact_levels"]) if stats["impact_levels"] else "No impact data"}
- Quality Scores (avg): Code {stats["avg_code_quality"]}/10, Best Practices {stats["avg_best_practices"]}/10,
Documentation {stats["avg_documentation"]}/10, Testing {stats["avg_testing"]}/10

**SAMPLE PR DETAILS:**
{json.dumps(pr_details, indent=2)}

**IDENTIFIED PATTERNS:**
Common Strengths: {
            ", ".join(set(stats["all_strengths"][:10]))
            if stats["all_strengths"]
            else "No specific strengths identified"
        }
Areas for Growth: {
            ", ".join(set(stats["all_improvements"][:10]))
            if stats["all_improvements"]
            else "No specific improvements identified"
        }

{previous_reports_context}

Create a balanced, professional performance review report that:
1. Highlights key accomplishments and impact
2. Provides specific examples of excellent work
3. Identifies patterns in code quality and practices
4. Offers constructive feedback for professional growth
5. Quantifies contribution and productivity
6. Assesses technical skills and collaboration
7. Evaluates the developer's experience level based on code patterns
8. Tracks improvement over time if previous reports are available

**OUTPUT FORMAT:**

# Performance Review: {author}
**Period:** {date_range}

## Executive Summary
[2-3 paragraph overview of performance, key contributions, and overall assessment]

## Experience Level Assessment
[Based on the code quality, architectural decisions, problem-solving approaches, and technical sophistication observed
in the PRs, assess whether this appears to be a junior (0-2 years), mid-level (2-5 years), senior (5+ years), or
staff-level developer. Consider factors like:
- Complexity of problems tackled
- Code organization and design patterns
- Error handling sophistication
- Testing approaches
- Documentation quality
- Ability to work independently vs. needing guidance]

## Quantitative Analysis
- **Productivity:** [Analysis of PR volume, code changes, and velocity]
- **Code Quality:** [Assessment based on scores and patterns]
- **Testing & Documentation:** [Evaluation of testing practices and documentation habits]

## Progress & Improvement
[If previous reports are available, analyze trends in code quality, types of work, and areas of growth. If no previous
reports, state "No previous reports available for comparison."]

## Key Accomplishments
[List 3-5 most significant contributions with specific impact]

## Technical Strengths
[Detailed analysis of demonstrated strengths with examples]

## Areas for Development
[Constructive feedback on areas for improvement with specific suggestions]

## Notable Contributions
[Highlight 2-3 PRs that best demonstrate the developer's capabilities]

## Primary Learning Recommendation
[Based on the patterns observed, identify ONE specific area of study or practice that would provide the highest ROI
for this developer's growth. Be specific - suggest actual resources, books, courses, or projects they could undertake.
Examples:
- "Study 'Design Patterns' by Gang of Four to improve architectural decisions"
- "Complete a course on advanced testing strategies, particularly around edge case identification"
- "Practice refactoring legacy code to improve code organization skills"
- "Study distributed systems concepts to better handle scalability challenges"]

## Additional Recommendations
[Other specific actions for continued growth and development]

Be specific, balanced, and constructive. Use data to support observations. Maximum 1200 words.
"""

        # Generate report with error handling
        try:
            response = self.ai_client.generate_with_retry(prompt, "best")
            if not response:
                logging.error(f"Failed to generate author report for {author}")
                return f"""# Performance Review: {author}
**Period:** {date_range}

## Error
Failed to generate AI-powered performance review.

### Summary Statistics:
- Total PRs: {stats["total_prs"]}
- Total Changes: +{stats["total_additions"]} / -{stats["total_deletions"]}
- Categories: {", ".join(f"{k}: {v}" for k, v in stats["categories"].items())}

Please try again or contact support if the issue persists.
"""

            return response

        except Exception as e:
            logging.error(f"Unexpected error generating report for {author}: {e}")
            return f"""# Performance Review: {author}
**Period:** {date_range}

## Error
An unexpected error occurred while generating the performance review: {str(e)}

Please check the logs and try again.
"""


class NewsletterGenerator:
    """Generates newsletter summaries from multiple PR summaries."""

    def __init__(self, ai_client: GeminiAIClient):
        self.ai_client = ai_client
        self.newsletter_extractor = PreviousReportExtractor(report_type="newsletter")

    def prepare_context(self, pr_summaries: list[PRSummary], date_range: str, repository: str) -> str:
        """Prepare context for newsletter generation."""
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

    def generate_newsletter(self, pr_summaries: list[PRSummary], date_range: str, repository: str) -> str:
        """Generate a comprehensive newsletter summary."""
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

Example of BAD Development Focus:
"This period marks a significant acceleration in our development velocity, characterized by a strategic push to
broaden the platform's applicability and appeal to the wider reinforcement learning community. While previous weeks
focused on solidifying our internal architecture, the theme now is external integration and advanced capability
expansion."

Example of GOOD Development Focus:
"We made MettaGrid compatible with Gymnasium, PettingZoo, and PufferLib (#1458), enabling use in standard RL
pipelines. Added systematic exploration memory configs (#1460), refactored the policy evaluation API with a new
client (#1488), and launched metta shell for interactive development (#1490). Codecov integration now tracks test
coverage (#1510)."

Why the GOOD Development Focus example is better:
- Lists specific deliverables with PR numbers
- Uses active voice and concrete verbs (made, added, refactored, launched)
- No abstract themes or philosophical language
- Every sentence contains actionable information
- Readers immediately understand what was built
- Avoids clichéd phrases like "this period" or "characterized by"

SHOUT OUT GUIDELINES:
- Only highlight truly exceptional work (1-3 maximum, or none if not warranted)
- IMPORTANT: Review previous newsletters and avoid repeatedly recognizing the same contributors
- If a developer received a shout out in recent newsletters, prioritize recognizing others who have done notable work
- Aim to distribute recognition across the entire team over time
- Focus on: complex problem-solving, code quality improvements, mentorship, critical fixes, or innovative solutions
- Format: "👏 @username - [specific achievement in 1-2 sentences]"

OUTPUT FORMAT:

## Summary of changes from {date_range}

**Development Focus:** [In 2-3 concise sentences, describe what the team actually built or fixed this period. Be
specific and direct - avoid abstract themes or philosophical descriptions. Lead with concrete work rather than
narrative descriptions. Vary your language across summaries.]

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
            newsletter_content,
            "",
            f"📦 [**Download Complete Analysis**]({github_run_url})",
        ]

        return "\n".join(lines)

    @staticmethod
    def create_author_report_output(
        pr_summaries: list[PRSummary],
        author_report: str,
        author: str,
        date_range: str,
        stats: dict,
    ) -> dict[str, str]:
        """Create author report outputs in multiple formats."""
        outputs = {}

        # Markdown report
        outputs["author_report_output.md"] = author_report

        # JSON summary
        summary_data = {
            "author": author,
            "period": date_range,
            "total_prs": stats["total_prs"],
            "total_additions": stats["total_additions"],
            "total_deletions": stats["total_deletions"],
            "total_files_changed": stats["total_files_changed"],
            "categories": dict(stats["categories"]),
            "impact_levels": dict(stats["impact_levels"]),
            "average_scores": {
                "code_quality": stats["avg_code_quality"],
                "best_practices": stats["avg_best_practices"],
                "documentation": stats["avg_documentation"],
                "testing": stats["avg_testing"],
            },
            "pr_list": [
                {
                    "number": pr.pr_number,
                    "title": pr.title,
                    "category": pr.category,
                    "impact": pr.impact_level,
                    "url": pr.html_url,
                }
                for pr in pr_summaries
            ],
        }

        outputs["author_report_summary.json"] = json.dumps(summary_data, indent=2)

        return outputs


class PRDigestAnalyzer:
    """Main orchestrator for analyzing PR digests."""

    def __init__(self, api_key: str):
        self.ai_client = GeminiAIClient(api_key)
        self.pr_analyzer = PRAnalyzer(self.ai_client)
        self.newsletter_generator = NewsletterGenerator(self.ai_client)
        self.author_report_generator = AuthorReportGenerator(self.ai_client)
        self.output_formatter = OutputFormatter()
        self.cached_loader = CachedSummaryLoader()

    def analyze_digest(
        self,
        new_prs: list[dict],
        cached_pr_numbers: list[int],
        use_parallel: bool = True,
        max_workers: int = 5,
        mode: str = "newsletter",
    ) -> list[PRSummary]:
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


def main():
    """Main entry point for PR digest analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check for mode flag
    report_mode = "newsletter"  # default
    if "--author-report" in sys.argv:
        report_mode = "author"
        logging.info("Running in AUTHOR REPORT mode")
    else:
        logging.info("Running in NEWSLETTER mode")

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
        "REPORT_TYPE": report_mode,
    }

    # Add author-specific requirements
    if report_mode == "author":
        required_vars.append("REPORT_AUTHOR")

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
    report_author = env_values.get("REPORT_AUTHOR")

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

        if report_mode == "newsletter":
            # Create minimal output files for newsletter
            with open("discord_summary_output.txt", "w") as f:
                f.write(f"📊 **PR Summary Report** • {report_period}\n\n")
                f.write("ℹ️ No PRs were merged during this period.\n")

            with open("newsletter_output.txt", "w") as f:
                f.write("No PRs to summarize in this period.")

            with open("pr_summary_data.json", "w") as f:
                json.dump([], f)
        else:
            # Create minimal output files for author report
            with open("author_report_output.md", "w") as f:
                f.write(f"# PR Report for @{report_author}\n\n")
                f.write(f"**Period**: {report_period}\n\n")
                f.write("No PRs were found for this author during the specified time period.")

            with open("author_report_summary.json", "w") as f:
                json.dump(
                    {
                        "author": report_author,
                        "period": report_period,
                        "total_prs": 0,
                        "message": "No PRs found for this author in the specified period",
                    },
                    f,
                )

            with open("pr_summary_data.json", "w") as f:
                json.dump([], f)

        print("Created minimal output files")
        sys.exit(0)

    cached_pr_numbers = stats.get("cached_pr_numbers", [])

    print("Processing PR digest:")
    print(f"  - Total PRs in period: {stats['total_prs_in_range']}")
    print(f"  - New PRs to analyze: {len(new_prs)}")
    print(f"  - Cached PRs to load: {len(cached_pr_numbers)}")
    if report_mode == "author":
        print(f"  - Author: {report_author}")

    # Initialize analyzer and process
    analyzer = PRDigestAnalyzer(api_key)
    all_summaries = analyzer.analyze_digest(new_prs, cached_pr_numbers, mode=report_mode)

    if not all_summaries:
        print("No summaries generated")
        sys.exit(1)

    print(f"✅ Processed {len(all_summaries)} total PR summaries")

    # Save individual PR files for newly processed PRs
    pr_summaries_dir = Path("pr-summaries")
    pr_summaries_dir.mkdir(exist_ok=True)

    new_summaries = [s for s in all_summaries if s.source == "new"]
    if new_summaries:
        print(f"Saving {len(new_summaries)} new PR summaries to {pr_summaries_dir}/...")
        for pr_summary in new_summaries:
            filepath = save_pr_summary(pr_summary, pr_summaries_dir)
            logging.debug(f"Saved {filepath}")
        print(f"✅ Saved {len(new_summaries)} new PR files")

    # Save structured data (all summaries)
    with open("pr_summary_data.json", "w") as f:
        json.dump([asdict(pr) for pr in all_summaries], f, indent=2)
    print("✅ Saved pr_summary_data.json")

    if report_mode == "newsletter":
        # Newsletter mode: Generate collection summary and Discord output
        pr_summaries = [s for s in all_summaries]

        print("Generating newsletter content...")
        newsletter_content = analyzer.newsletter_generator.generate_newsletter(
            pr_summaries, report_period, github_repository
        )

        with open("newsletter_output.txt", "w") as f:
            f.write(newsletter_content)
        print("✅ Saved newsletter_output.txt")

        # Create Discord-formatted output
        print("Generating Discord summary...")
        discord_summary = analyzer.output_formatter.create_discord_summary(
            pr_summaries, newsletter_content, report_period, github_run_url, stats, github_repository
        )

        with open("discord_summary_output.txt", "w") as f:
            f.write(discord_summary)
        print("✅ Saved discord_summary_output.txt")

    else:
        # Author report mode: Generate author-specific report
        assert report_author is not None

        print(f"Generating author report for {report_author}...")

        author_stats = analyzer.author_report_generator.calculate_author_stats(all_summaries)
        author_report = analyzer.author_report_generator.generate_author_report(
            all_summaries, report_period, report_author
        )

        # Create outputs
        outputs = analyzer.output_formatter.create_author_report_output(
            all_summaries, author_report, report_author, report_period, author_stats
        )

        for filename, content in outputs.items():
            with open(filename, "w") as f:
                f.write(content)
            print(f"✅ Saved {filename}")

    print("🎉 PR digest analysis complete!")


if __name__ == "__main__":
    main()
