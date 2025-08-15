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
from pathlib import Path

from gemini_analyze_pr import PRSummary
from gemini_analyze_pr_digest import (
    PRDigestAnalyzer,
    load_digest_data,
    save_new_summaries,
    save_structured_data,
)
from gemini_client import GeminiAIClient


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

    def get_previous_reports_context(self, author: str) -> str:
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

    def get_bonus_prompt(self) -> str:
        prompts = [
            "Based on this author's PRs in the last 30 days, they were likely distracted by...",
            "My best guess at this developer's hobbies and interests based solely on their commit messages:",
            "Write a few stanzas of an epic poem about the author's battles with legacy code.",
            "A motivational quote that would resonate deeply with this developer:",
            "What mythical creature best represents this developer's coding style, and why?",
            "A short fable in which the author learns a lesson about code review.",
            "A LinkedIn endorsement written by a medieval bard.",
            "Imagine this author is a character in a fantasy novel. Describe their class, abilities, and greatest feat.",
            "Draft the first lines of a documentary about this developer’s journey through this review period.",
            "Write a single tweet that encapsulates this author's coding spirit.",
        ]
        return random.choice(prompts)

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

        previous_context = self.get_previous_reports_context(author)

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

{previous_context}

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

**Closing Thoughts:**
[{self.get_bonus_prompt()}]

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


def main():
    """Main entry point for author report generation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Running in AUTHOR REPORT mode")

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
        "REPORT_AUTHOR",  # Required for author reports
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
    pr_digest_file = env_values["PR_DIGEST_FILE"]
    stats_file = env_values["PR_DIGEST_STATS_FILE"]
    report_period = env_values["REPORT_PERIOD"]
    report_author = env_values["REPORT_AUTHOR"]

    try:
        # Load PR digest and stats
        new_prs, stats = load_digest_data(pr_digest_file, stats_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Check if there are any PRs in the time period
    if stats["total_prs_in_range"] == 0:
        print("No PRs found in the specified time period")

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
    print(f"  - Author: {report_author}")

    # Initialize analyzer and process
    analyzer = PRDigestAnalyzer(api_key)
    all_summaries = analyzer.analyze_digest(new_prs, cached_pr_numbers)

    # Filter summaries for the specific author
    author_summaries = [s for s in all_summaries if s.author.lower() == report_author.lower()]

    if not author_summaries:
        print(f"No summaries found for author: {report_author}")

        # Create minimal output files
        with open("author_report_output.md", "w") as f:
            f.write(f"# PR Report for @{report_author}\n\n")
            f.write(f"**Period**: {report_period}\n\n")
            f.write(f"No PRs by {report_author} were found in this period.")

        with open("author_report_summary.json", "w") as f:
            json.dump(
                {
                    "author": report_author,
                    "period": report_period,
                    "total_prs": 0,
                    "message": f"No PRs found for {report_author}",
                },
                f,
            )

        with open("pr_summary_data.json", "w") as f:
            json.dump([], f)

        sys.exit(0)

    print(f"✅ Found {len(author_summaries)} PRs by {report_author}")

    # Save individual PR files for newly processed PRs
    new_count = save_new_summaries(all_summaries)
    if new_count > 0:
        print(f"✅ Saved {new_count} new PR files")

    # Save structured data (author's summaries only)
    save_structured_data(author_summaries)
    print("✅ Saved pr_summary_data.json")

    # Generate author report
    print(f"Generating author report for {report_author}...")
    ai_client = GeminiAIClient(api_key)
    author_report_generator = AuthorReportGenerator(ai_client)

    author_stats = author_report_generator.calculate_author_stats(author_summaries)
    author_report = author_report_generator.generate_author_report(author_summaries, report_period, report_author)

    # Create outputs

    outputs = create_author_report_output(author_summaries, author_report, report_author, report_period, author_stats)

    for filename, content in outputs.items():
        with open(filename, "w") as f:
            f.write(content)
        print(f"✅ Saved {filename}")

    print(f"🎉 Author report for {report_author} complete!")


if __name__ == "__main__":
    main()
