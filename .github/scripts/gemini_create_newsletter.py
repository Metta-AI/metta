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
from datetime import datetime
from pathlib import Path
from typing import Optional

from gemini_analyze_pr import PRSummary
from gemini_analyze_pr_digest import (
    PRDigestAnalyzer,
    PreviousReportExtractor,
    create_discord_summary,
    load_digest_data,
    save_new_summaries,
    save_structured_data,
)
from gemini_client import GeminiAIClient


class NewsletterGenerator:
    """Generates newsletter summaries from multiple PR summaries."""

    def __init__(self, ai_client: GeminiAIClient, is_historical: bool = False):
        self.ai_client = ai_client
        self.is_historical = is_historical
        self.newsletter_extractor = PreviousReportExtractor(report_type="newsletter")

    def get_previous_newsletter_context(self, end_date: Optional[datetime] = None) -> str:
        """Format newsletter summaries for context."""
        # For historical runs, pass the end date to filter out future newsletters
        recent_summaries = self.newsletter_extractor.get_recent_summaries(
            end_date=end_date if self.is_historical else None
        )

        all_shout_outs = []
        for summary in recent_summaries:
            content = summary["content"]
            if "**Shout Outs:**" in content:
                shout_section = content.split("**Shout Outs:**")[1].split("**")[0]
                import re

                mentioned_users = re.findall(r"🎉 @(\w+)", shout_section)
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

    def get_bonus_prompt(self) -> str:
        bonus_prompts = [
            "Write a zen koan reflecting on code review.",
            "Generate a haiku summarizing this week's PRs.",
            "Ask a rhetorical question about productivity.",
            "Invent a proverb inspired by Git.",
            "Pretend you are an oracle and give a cryptic message.",
        ]
        return random.choice(bonus_prompts)

    def generate_newsletter(
        self, pr_summaries: list[PRSummary], date_range: str, repository: str, end_date: Optional[datetime] = None
    ) -> str:
        """Generate a comprehensive newsletter summary."""
        context = self.prepare_context(pr_summaries, date_range, repository)
        previous_context = self.get_previous_newsletter_context(end_date)

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

**Closing Thoughts:** [{self.get_bonus_prompt()}]
"""

        ai_response = self.ai_client.generate_with_retry(prompt, "best")

        if not ai_response:
            logging.error("AI generation failed for collection summary")
            return "Failed to generate collection summary"

        return ai_response


def main():
    """Main entry point for newsletter generation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        "IS_HISTORICAL_RUN": "false",
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
    is_historical = env_values["IS_HISTORICAL_RUN"].lower() == "true"

    # Log if this is a historical run
    if is_historical:
        logging.info(f"Generating newsletter for: {report_period}")

    # Construct GitHub run URL
    github_run_url = f"{github_server_url}/{github_repository}/actions/runs/{github_run_id}"

    try:
        # Load PR digest and stats
        new_prs, stats = load_digest_data(pr_digest_file, stats_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Check if there are any PRs in the time period
    if stats["total_prs_in_range"] == 0:
        print("No PRs found in the specified time period")

        # Create minimal output files
        with open("discord_summary_output.txt", "w") as f:
            f.write(f"📊 **PR Summary Report** • {report_period}\n\n")
            f.write("ℹ️ No PRs were merged during this period.\n")

        with open("newsletter_output.txt", "w") as f:
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

    # Save individual PR files for newly processed PRs
    new_count = save_new_summaries(all_summaries)
    if new_count > 0:
        print(f"✅ Saved {new_count} new PR files")

    # Save structured data (all summaries)
    save_structured_data(all_summaries)
    print("✅ Saved pr_summary_data.json")

    # Generate newsletter content
    print("Generating newsletter content...")
    ai_client = GeminiAIClient(api_key)
    newsletter_generator = NewsletterGenerator(ai_client, is_historical)

    # Parse end date for historical context filtering
    end_date = None
    if is_historical:
        # Get the end date from the stats file
        end_date_str = stats.get("end_date")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
            except ValueError:
                logging.warning(f"Could not parse end date from stats: {end_date_str}")

    newsletter_content = newsletter_generator.generate_newsletter(
        all_summaries, report_period, github_repository, end_date
    )

    with open("newsletter_output.txt", "w") as f:
        f.write(newsletter_content)
    print("✅ Saved newsletter_output.txt")

    # Create Discord-formatted output
    print("Generating Discord summary...")

    discord_summary = create_discord_summary(
        all_summaries, newsletter_content, report_period, github_run_url, stats, github_repository
    )

    with open("discord_summary_output.txt", "w") as f:
        f.write(discord_summary)
    print("✅ Saved discord_summary_output.txt")

    print("🎉 Newsletter generation complete!")


if __name__ == "__main__":
    main()
