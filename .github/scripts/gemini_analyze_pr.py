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
from typing import Dict, List, Optional

from gemini_client import GeminiAIClient


@dataclass
class PRSummary:
    """Individual PR summary with essential metadata."""

    pr_number: int
    title: str
    author: str
    merged_at: str
    html_url: str
    summary: str
    key_changes: List[str]
    developer_impact: str
    technical_notes: str
    impact_level: str  # "minor", "moderate", "major"
    category: str  # "feature", "bugfix", "docs", "refactor", etc.


class PRAnalyzer:
    """Analyzes a single PR and generates a comprehensive summary."""

    def __init__(self, ai_client: GeminiAIClient):
        self.ai_client = ai_client

    def categorize_pr(self, pr_data: Dict) -> str:
        """Enhanced PR categorization based on labels, title, and content."""
        title = pr_data.get("title", "").lower()
        labels = [label.lower() for label in pr_data.get("labels", [])]
        body = pr_data.get("body", "").lower()

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
        }

        for pattern_group, category in title_patterns.items():
            if any(word in title for word in pattern_group):
                return category

        # Check body for issue references
        if any(phrase in body for phrase in ["fixes #", "closes #", "resolves #"]):
            return "bugfix"

        return "other"

    def estimate_impact_level(self, pr_data: Dict) -> str:
        """Estimate the impact level based on changes and content."""
        diff = pr_data.get("diff", "")
        title = pr_data.get("title", "").lower()
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

    def create_analysis_prompt(self, pr_data: Dict) -> str:
        """Create a comprehensive analysis prompt for the PR."""
        diff = pr_data.get("diff", "")
        description = pr_data.get("body", "No description provided")

        # Extract file information
        file_changes = re.findall(r"^diff --git a/(.+?) b/(.+?)$", diff, re.MULTILINE)
        file_types = set()
        for _old_file, new_file in file_changes:
            if "." in new_file:
                file_types.add(new_file.split(".")[-1])

        estimated_tokens = len(diff + description) // 4
        logging.info(f"PR #{pr_data['number']}: Using ~{estimated_tokens:,} tokens")

        prompt = f"""
Analyze this Pull Request comprehensively for an engineering audience.

**PR #{pr_data["number"]}: {pr_data["title"]}**
- Author: {pr_data["author"]}
- Labels: {", ".join(pr_data.get("labels", [])) if pr_data.get("labels") else "None"}
- Files: {len(file_changes)} files, types: {", ".join(sorted(file_types)) if file_types else "N/A"}
- Changes: {pr_data.get("additions", 0)} additions, {pr_data.get("deletions", 0)} deletions

**Description:**
{description}

**Code Changes:**
{diff}

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

Maximum 500 words total. Be comprehensive within this limit.
"""
        return prompt

    def parse_ai_response(self, response: str) -> Dict:
        """Parse the AI response into structured data."""
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

    def analyze(self, pr_data: Dict, model_tier: str = "default") -> Optional[Dict]:
        """Analyze a single PR and return structured summary data."""
        pr_number = pr_data["number"]
        logging.info(f"Analyzing PR #{pr_number}: {pr_data['title']}")

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

        # Return structured data
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
        }


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
