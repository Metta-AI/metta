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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from gemini_client import MODEL_CONFIG, GeminiAIClient


@dataclass
class PRSummary:
    """Complete PR summary with metadata and quality assessments."""

    # Core fields
    pr_number: int
    title: str
    author: str
    merged_at: str
    html_url: str
    summary: str
    key_changes: list[str]
    developer_impact: str
    technical_notes: Optional[str]
    impact_level: str
    category: str

    # Quality assessments
    code_quality_score: Optional[int] = None
    code_quality_notes: Optional[str] = None
    best_practices_score: Optional[int] = None
    best_practices_notes: Optional[str] = None
    documentation_score: Optional[int] = None
    documentation_notes: Optional[str] = None
    testing_score: Optional[int] = None
    testing_notes: Optional[str] = None
    strengths: Optional[list[str]] = None
    areas_for_improvement: Optional[list[str]] = None

    # Statistics
    additions: Optional[int] = None
    deletions: Optional[int] = None
    files_changed: Optional[int] = None
    test_coverage_impact: Optional[str] = None
    review_cycles: Optional[int] = None

    # Metadata
    source: Literal["cache", "new"] = "new"
    labels: Optional[list[str]] = None
    draft: bool = False


class PRAnalyzer:
    """Analyzes a single PR and generates a comprehensive summary."""

    def __init__(self, ai_client: GeminiAIClient):
        self.ai_client = ai_client

    def categorize_pr(self, pr_data: dict) -> str:
        """Enhanced PR categorization based on labels, title, and content."""
        title = (pr_data.get("title") or "").lower()
        labels = [label.lower() for label in pr_data.get("labels", [])]
        body = (pr_data.get("body") or "").lower()

        # Check labels first (most reliable)
        label_mapping = {
            ("bug", "bugfix", "fix"): "bugfix",
            ("feature", "enhancement"): "feature",
            ("docs", "documentation"): "documentation",
            ("refactor", "cleanup"): "refactor",
            ("test", "testing"): "testing",
            ("ci", "build", "deployment"): "infrastructure",
            ("security",): "security",
            ("dependency", "dependencies"): "dependency",
            ("performance", "optimization"): "performance",
        }

        for label_group, category in label_mapping.items():
            if any(label in label_group for label in labels):
                return category

        # Check title patterns
        title_patterns = {
            ("fix", "bug", "error", "resolve"): "bugfix",
            ("add", "new", "feature", "implement"): "feature",
            ("doc", "readme", "guide"): "documentation",
            ("refactor", "clean", "improve", "optimize"): "refactor",
            ("test", "spec", "coverage"): "testing",
            ("bump", "update", "upgrade", "dependency"): "dependency",
            ("security", "vulnerability", "cve"): "security",
            ("perf", "performance", "speed", "optimize"): "performance",
        }

        for pattern_group, category in title_patterns.items():
            if any(word in title for word in pattern_group):
                return category

        # Check body for issue references
        if any(phrase in body for phrase in ["fixes #", "closes #", "resolves #"]):
            return "bugfix"

        return "other"

    def estimate_impact_level(self, pr_data: dict) -> str:
        """Estimate the impact level based on changes and content."""
        diff = pr_data.get("diff") or ""
        title = (pr_data.get("title") or "").lower()
        labels = [label.lower() for label in pr_data.get("labels", [])]

        lines_added = pr_data.get("additions", 0)
        lines_deleted = pr_data.get("deletions", 0)
        total_lines = lines_added + lines_deleted
        files_changed = pr_data.get("changed_files", 0)

        # Check for explicit breaking change indicators
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
            r"interface\s+\w+",  # New interfaces
            r"public\s+.*\s+class",  # Public class changes
        ]

        critical_matches = sum(1 for pattern in critical_patterns if re.search(pattern, diff, re.IGNORECASE))

        if critical_matches > 0:
            if total_lines > 100 or files_changed > 5:
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

    def analyze_test_coverage(self, diff: str) -> str:
        """Analyze if PR includes tests or affects test coverage."""
        test_patterns = [
            r"test_.*\.py",
            r".*_test\.py",
            r"spec\..*",
            r".*\.spec\.",
            r"test/",
            r"tests/",
            r"__tests__/",
        ]

        has_test_files = any(re.search(pattern, diff, re.IGNORECASE) for pattern in test_patterns)

        # Check for test additions
        test_additions = re.findall(r"\+.*(?:test|spec|assert|expect)", diff, re.IGNORECASE)
        test_deletions = re.findall(r"-.*(?:test|spec|assert|expect)", diff, re.IGNORECASE)

        if has_test_files and len(test_additions) > len(test_deletions):
            return "increased"
        elif has_test_files and len(test_additions) < len(test_deletions):
            return "decreased"
        elif has_test_files:
            return "maintained"
        else:
            return "unknown"

    def create_analysis_prompt(self, pr_data: dict) -> str:
        """Create analysis prompt based on the mode (newsletter or author_report)."""

        # Common header for both modes
        diff = pr_data.get("diff") or ""
        description = pr_data.get("body") or "No description provided"

        # Extract file information
        file_changes = re.findall(r"^diff --git a/(.+?) b/(.+?)$", diff, re.MULTILINE)
        file_types = set()
        for _old_file, new_file in file_changes:
            if "." in new_file:
                file_types.add(new_file.split(".")[-1])

        estimated_tokens = len(diff + description) // 4
        logging.info(f"PR #{pr_data['number']}: Using ~{estimated_tokens:,} tokens")

        header = f"""
    Analyze this Pull Request comprehensively.

    **PR #{pr_data["number"]}: {pr_data["title"]}**
    - Author: {pr_data["author"]}
    - Labels: {", ".join(pr_data.get("labels", [])) if pr_data.get("labels") else "None"}
    - Files: {len(file_changes)} files, types: {", ".join(sorted(file_types)) if file_types else "N/A"}
    - Changes: {pr_data.get("additions", 0)} additions, {pr_data.get("deletions", 0)} deletions

    **Description:**
    {description}

    **Code Changes:**
    {diff}
    """

        return self._create_prompt(header)

    def _create_prompt(self, header: str) -> str:
        """Create prompt for newsletter-style summary."""
        return (
            header
            + """
    **Required Output Format for analysis:**

    **Summary:** [2-3 sentences explaining what this PR accomplishes, why it was needed, and its primary benefit]

    **Key Changes:**
    - [Most significant functional or architectural change with specific details]
    - [Second most important change - focus on implementation or API changes]
    - [Third change if applicable - additional notable modifications]

    **Analysis Guidelines:**
    - Focus on technical facts and engineering impact
    - Be concise and direct
    - Emphasize what was built and how it works
    - Skip subjective quality assessments
    - Maximum 500 words total

    **Developer Impact:** [Concrete effects on other developers: new/changed APIs, breaking changes, new dependencies,
    modified workflows, testing requirements. If truly minimal, state "Minimal developer impact."]

    **Technical Notes:** [Implementation details, architectural decisions, performance implications, trade-offs, or
    gotchas worth highlighting]

    **Code Quality Assessment:**
    - **Score (1-10):** [Evaluate readability, maintainability, design patterns, error handling]
    - **Notes:** [Specific observations about code quality, both positive and negative]

    **Best Practices Assessment:**
    - **Score (1-10):** [Evaluate adherence to coding standards, security practices, performance considerations]
    - **Notes:** [Specific observations about best practices followed or violated]

    **Documentation Assessment:**
    - **Score (1-10):** [Evaluate code comments, docstrings, README updates, API documentation]
    - **Notes:** [Specific observations about documentation quality and completeness]

    **Testing Assessment:**
    - **Score (1-10):** [Evaluate test coverage, test quality, edge cases handled]
    - **Notes:** [Specific observations about testing approach and thoroughness]

    **Strengths:** [List 2-3 specific technical strengths demonstrated in this PR]
    - [Strength 1 - be specific]
    - [Strength 2 - be specific]
    - [Strength 3 if applicable]

    **Areas for Improvement:** [List 2-3 specific areas where the code could be improved]
    - [Area 1 - be constructive and specific]
    - [Area 2 - be constructive and specific]
    - [Area 3 if applicable]

    **Analysis Guidelines:**
    - Be objective and constructive in assessments
    - Focus on concrete examples from the code
    - Consider the context and constraints
    - Highlight both positive aspects and areas for growth
    - Scores should reflect: 1-3 (needs significant improvement), 4-6 (adequate), 7-8 (good), 9-10 (excellent)
    - Be fair but honest in evaluations
    - Maximum 800 words total. Be comprehensive and balanced.
    """
        )

    def parse_ai_response(self, response: str) -> dict:
        """Parse the AI response into structured data."""
        parsed = {
            "summary": "",
            "key_changes": [],
            "developer_impact": "",
            "technical_notes": "",
            "code_quality_score": 5,
            "code_quality_notes": "",
            "best_practices_score": 5,
            "best_practices_notes": "",
            "documentation_score": 5,
            "documentation_notes": "",
            "testing_score": 5,
            "testing_notes": "",
            "strengths": [],
            "areas_for_improvement": [],
        }

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

        # Extract assessments
        assessments = [
            ("Code Quality", "code_quality"),
            ("Best Practices", "best_practices"),
            ("Documentation", "documentation"),
            ("Testing", "testing"),
        ]

        for assessment_name, field_prefix in assessments:
            section = re.search(
                (
                    rf"\*\*{assessment_name} Assessment:\*\*\s*"
                    r"(?:.*?)"
                    r"\*\*Score.*?:\*\*\s*"
                    r"(\d+).*?"
                    r"\*\*Notes:\*\*\s*"
                    r"(.+?)"
                    r"(?=\*\*|$)"
                ),
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if section:
                try:
                    score = int(section.group(1))
                    parsed[f"{field_prefix}_score"] = max(1, min(10, score))
                except ValueError:
                    parsed[f"{field_prefix}_score"] = 5
                parsed[f"{field_prefix}_notes"] = section.group(2).strip()

        # Extract strengths
        strengths_section = re.search(r"\*\*Strengths:\*\*\s*((?:•.*?(?:\n|$))+)", response, re.DOTALL)
        if strengths_section:
            strengths_text = strengths_section.group(1)
            parsed["strengths"] = [
                strength.strip().lstrip("•").strip()
                for strength in strengths_text.split("\n")
                if strength.strip() and "•" in strength
            ]

        # Extract areas for improvement
        areas_section = re.search(r"\*\*Areas for Improvement:\*\*\s*((?:•.*?(?:\n|$))+)", response, re.DOTALL)
        if areas_section:
            areas_text = areas_section.group(1)
            parsed["areas_for_improvement"] = [
                area.strip().lstrip("•").strip() for area in areas_text.split("\n") if area.strip() and "•" in area
            ]

        return parsed

    def analyze(self, pr_data: dict, model_tier: str = "default") -> Optional[dict]:
        """Analyze a single PR and return structured summary data."""
        pr_number = pr_data["number"]
        logging.info(f"Analyzing PR #{pr_number}: {pr_data['title']}")

        # sanitize pr_data
        pr_data["title"] = pr_data.get("title") or "Untitled"
        pr_data["author"] = pr_data.get("author") or "Unknown"
        pr_data["merged_at"] = pr_data.get("merged_at") or ""
        pr_data["html_url"] = pr_data.get("html_url") or ""

        # Create analysis prompt
        prompt = self.create_analysis_prompt(pr_data)

        # Generate analysis using specified model tier
        ai_response = self.ai_client.generate_with_retry(prompt, model_tier)

        if not ai_response:
            logging.error(f"AI generation failed for PR #{pr_number}")
            return None

        # Parse the response
        parsed = self.parse_ai_response(ai_response)

        # Determine category and impact level
        category = self.categorize_pr(pr_data)
        impact_level = self.estimate_impact_level(pr_data)
        test_coverage_impact = self.analyze_test_coverage(pr_data.get("diff", ""))

        # Return structured data with all fields
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
            # Quality assessments
            "code_quality_score": parsed["code_quality_score"],
            "code_quality_notes": parsed["code_quality_notes"],
            "best_practices_score": parsed["best_practices_score"],
            "best_practices_notes": parsed["best_practices_notes"],
            "documentation_score": parsed["documentation_score"],
            "documentation_notes": parsed["documentation_notes"],
            "testing_score": parsed["testing_score"],
            "testing_notes": parsed["testing_notes"],
            # Feedback
            "strengths": parsed["strengths"],
            "areas_for_improvement": parsed["areas_for_improvement"],
            # Statistics
            "additions": pr_data.get("additions", 0),
            "deletions": pr_data.get("deletions", 0),
            "files_changed": pr_data.get("changed_files", 0),
            "test_coverage_impact": test_coverage_impact,
            "review_cycles": pr_data.get("review_comments", 0),  # Approximate from review comments
            # Additional metadata
            "labels": pr_data.get("labels", []),
            "draft": pr_data.get("draft", False),
        }


def save_pr_summary(pr_summary: PRSummary, output_dir: Path) -> str:
    """Save individual PR summary as a text file with quality metrics if available."""
    filename = f"pr_{pr_summary.pr_number}.txt"
    filepath = output_dir / filename

    content = f"""PR #{pr_summary.pr_number}: {pr_summary.title}

Author: {pr_summary.author}
Merged: {pr_summary.merged_at}
Category: {pr_summary.category.title()}
Impact: {pr_summary.impact_level.title()}
GitHub: {pr_summary.html_url}
"""

    # Add statistics if available
    if pr_summary.additions is not None:
        content += f"Additions: {pr_summary.additions}\n"
    if pr_summary.deletions is not None:
        content += f"Deletions: {pr_summary.deletions}\n"
    if pr_summary.files_changed is not None:
        content += f"Files Changed: {pr_summary.files_changed}\n"
    if pr_summary.review_cycles is not None:
        content += f"Review Cycles: {pr_summary.review_cycles}\n"
        content += """
================================================================================

SUMMARY
{summary}

KEY CHANGES
{changes}

DEVELOPER IMPACT
{impact}

TECHNICAL NOTES
{notes}
""".format(
            summary=pr_summary.summary,
            changes=chr(10).join(f"• {change}" for change in pr_summary.key_changes),
            impact=pr_summary.developer_impact,
            notes=pr_summary.technical_notes if pr_summary.technical_notes else "No additional technical notes",
        )

    # Add quality assessments if available
    if pr_summary.code_quality_score is not None:
        content += f"""
CODE QUALITY
Score: {pr_summary.code_quality_score}/10
{pr_summary.code_quality_notes}

BEST PRACTICES
Score: {pr_summary.best_practices_score}/10
{pr_summary.best_practices_notes}

DOCUMENTATION
Score: {pr_summary.documentation_score}/10
{pr_summary.documentation_notes}

TESTING
Score: {pr_summary.testing_score}/10
{pr_summary.testing_notes}
"""

    if pr_summary.strengths:
        content += f"""
STRENGTHS
{chr(10).join(f"• {strength}" for strength in pr_summary.strengths)}
"""

    if pr_summary.areas_for_improvement:
        content += f"""
AREAS FOR IMPROVEMENT
{chr(10).join(f"• {area}" for area in pr_summary.areas_for_improvement)}
"""

    content += f"""
================================================================================
Generated using {MODEL_CONFIG["default"]} on {datetime.now().isoformat()}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return str(filepath)


def load_pr_summary(pr_number: int, summaries_dir: Path) -> Optional[PRSummary]:
    """Load a single PR summary from cache."""
    summary_file = summaries_dir / f"pr_{pr_number}.txt"

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
            # Quality metrics (may not exist in older cached files)
            "code_quality_score": None,
            "code_quality_notes": "",
            "best_practices_score": None,
            "best_practices_notes": "",
            "documentation_score": None,
            "documentation_notes": "",
            "testing_score": None,
            "testing_notes": "",
            "strengths": [],
            "areas_for_improvement": [],
            "additions": None,
            "deletions": None,
            "files_changed": None,
            "test_coverage_impact": "",
            "review_cycles": None,  # Add this field
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
            elif line.startswith("Additions: "):
                try:
                    sections["additions"] = int(line.replace("Additions: ", "").strip())
                except ValueError:
                    pass
            elif line.startswith("Deletions: "):
                try:
                    sections["deletions"] = int(line.replace("Deletions: ", "").strip())
                except ValueError:
                    pass
            elif line.startswith("Files Changed: "):
                try:
                    sections["files_changed"] = int(line.replace("Files Changed: ", "").strip())
                except ValueError:
                    pass
            elif line.startswith("Review Cycles: "):  # Add this parser
                try:
                    sections["review_cycles"] = int(line.replace("Review Cycles: ", "").strip())
                except ValueError:
                    pass

            # Section markers
            elif line.strip() == "SUMMARY":
                current_section = "summary"
            elif line.strip() == "KEY CHANGES":
                current_section = "key_changes"
            elif line.strip() == "DEVELOPER IMPACT":
                current_section = "developer_impact"
            elif line.strip() == "TECHNICAL NOTES":
                current_section = "technical_notes"
            elif line.strip() == "CODE QUALITY":
                current_section = "code_quality"
            elif line.strip() == "BEST PRACTICES":
                current_section = "best_practices"
            elif line.strip() == "DOCUMENTATION":
                current_section = "documentation"
            elif line.strip() == "TESTING":
                current_section = "testing"
            elif line.strip() == "STRENGTHS":
                current_section = "strengths"
            elif line.strip() == "AREAS FOR IMPROVEMENT":
                current_section = "areas_for_improvement"
            elif line.startswith("=" * 20):
                current_section = None

            # Content within sections
            elif current_section and line.strip():
                if current_section == "key_changes" and line.startswith("• "):
                    sections["key_changes"].append(line[2:].strip())
                elif current_section == "strengths" and line.startswith("• "):
                    sections["strengths"].append(line[2:].strip())
                elif current_section == "areas_for_improvement" and line.startswith("• "):
                    sections["areas_for_improvement"].append(line[2:].strip())
                elif current_section in ["summary", "developer_impact", "technical_notes"]:
                    if sections[current_section]:
                        sections[current_section] += " "
                    sections[current_section] += line.strip()
                elif current_section in ["code_quality", "best_practices", "documentation", "testing"]:
                    if line.startswith("Score: "):
                        try:
                            score = int(line.replace("Score: ", "").strip())
                            sections[f"{current_section}_score"] = score
                        except ValueError:
                            pass
                    else:
                        if sections[f"{current_section}_notes"]:
                            sections[f"{current_section}_notes"] += " "
                        sections[f"{current_section}_notes"] += line.strip()

        # Create PRSummary object
        return PRSummary(
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
            # Quality metrics
            code_quality_score=sections["code_quality_score"],
            code_quality_notes=sections["code_quality_notes"],
            best_practices_score=sections["best_practices_score"],
            best_practices_notes=sections["best_practices_notes"],
            documentation_score=sections["documentation_score"],
            documentation_notes=sections["documentation_notes"],
            testing_score=sections["testing_score"],
            testing_notes=sections["testing_notes"],
            strengths=sections["strengths"],
            areas_for_improvement=sections["areas_for_improvement"],
            additions=sections["additions"],
            deletions=sections["deletions"],
            files_changed=sections["files_changed"],
            test_coverage_impact=sections["test_coverage_impact"],
            review_cycles=sections["review_cycles"],
        )

    except Exception as e:
        logging.error(f"Error loading cached summary for PR #{pr_number}: {e}")
        return None


def main():
    """Analyze a single PR from command line arguments or stdin."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Handle different input methods
    if len(sys.argv) == 3:
        # Command line: api_key pr_json_string
        api_key = sys.argv[1]
        pr_json = sys.argv[2]
    elif len(sys.argv) == 2:
        # API key provided, read PR data from stdin
        api_key = sys.argv[1]
        pr_json = sys.stdin.read().strip()
    else:
        print("Usage: python gemini_analyze_pr.py <api_key> [pr_json]")
        print("       python gemini_analyze_pr.py <api_key> < pr_data.json")
        sys.exit(1)

    try:
        pr_data = json.loads(pr_json)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON: {e}")
        sys.exit(1)

    # Initialize client and analyzer
    ai_client = GeminiAIClient(api_key)
    analyzer = PRAnalyzer(ai_client)

    # Analyze the PR
    result = analyzer.analyze(pr_data)

    if result:
        print(json.dumps(result, indent=2))
    else:
        logging.error("Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
