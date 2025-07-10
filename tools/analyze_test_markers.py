#!/usr/bin/env python3
"""
Analyze test files to identify candidates for hourly/daily markers.

This script helps identify:
1. Critical tests that should run hourly
2. Integration tests that should run daily
3. Tests currently without schedule markers
"""

import ast
import re
from pathlib import Path
from typing import Dict, List


class TestAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze test functions and their characteristics."""

    def __init__(self):
        self.tests = []
        self.current_markers = set()
        self.current_class = None

    def visit_ClassDef(self, node):
        """Track test classes."""
        if node.name.startswith("Test"):
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze test functions."""
        if node.name.startswith("test_"):
            # Extract markers
            markers = set()
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if isinstance(decorator.value, ast.Attribute) and decorator.value.attr == "mark":
                        markers.add(decorator.attr)

            # Analyze test characteristics
            test_info = {
                "name": node.name,
                "class": self.current_class,
                "markers": markers,
                "lines": node.end_lineno - node.lineno + 1,
                "has_fixtures": self._uses_fixtures(node),
                "likely_integration": self._is_likely_integration(node),
                "likely_critical": self._is_likely_critical(node),
            }

            self.tests.append(test_info)

        self.generic_visit(node)

    def _uses_fixtures(self, node) -> bool:
        """Check if test uses fixtures."""
        return len(node.args.args) > 1  # More than just 'self'

    def _is_likely_integration(self, node) -> bool:
        """Heuristically determine if test is integration-like."""
        code = ast.unparse(node)
        integration_patterns = [
            r"wandb\.",
            r"s3\.",
            r"boto3",
            r"requests\.",
            r"httpx\.",
            r"docker",
            r"testcontainers",
            r"database",
            r"postgres",
            r"redis",
            r"api_client",
            r"external_service",
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in integration_patterns)

    def _is_likely_critical(self, node) -> bool:
        """Heuristically determine if test is critical path."""
        name_lower = node.name.lower()
        critical_patterns = [
            "init",
            "config",
            "setup",
            "core",
            "basic",
            "essential",
            "critical",
            "validation",
            "sanity",
        ]
        return any(pattern in name_lower for pattern in critical_patterns)


def analyze_test_file(file_path: Path) -> List[Dict]:
    """Analyze a single test file."""
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        analyzer = TestAnalyzer()
        analyzer.visit(tree)

        for test in analyzer.tests:
            test["file"] = str(file_path)

        return analyzer.tests
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def categorize_tests(tests: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize tests based on their characteristics."""
    categories = {
        "should_be_hourly": [],
        "should_be_daily": [],
        "already_marked": [],
        "unclassified": [],
    }

    for test in tests:
        markers = test["markers"]

        # Skip if already marked
        if any(m in markers for m in ["hourly", "daily", "slow"]):
            categories["already_marked"].append(test)
            continue

        # Classify based on characteristics
        if test["likely_critical"] and test["lines"] < 50:
            categories["should_be_hourly"].append(test)
        elif test["likely_integration"] or "integration" in markers:
            categories["should_be_daily"].append(test)
        elif test["lines"] > 100:
            categories["should_be_daily"].append(test)
        else:
            categories["unclassified"].append(test)

    return categories


def generate_recommendations(categories: Dict[str, List[Dict]]) -> None:
    """Generate recommendations for test markers."""
    print("# Test Marker Recommendations\n")

    print("## Tests that should be marked @pytest.mark.hourly")
    print("(Critical path tests that should run every hour)\n")

    if categories["should_be_hourly"]:
        for test in sorted(categories["should_be_hourly"], key=lambda t: t["file"]):
            rel_path = Path(test["file"]).relative_to(Path.cwd())
            print(f"- `{rel_path}::{test['name']}` - {test['lines']} lines")
            if test["likely_critical"]:
                print("  - Reason: Name suggests critical functionality")
    else:
        print("*No additional tests identified for hourly execution*")

    print("\n## Tests that should be marked @pytest.mark.daily")
    print("(Integration or comprehensive tests)\n")

    if categories["should_be_daily"]:
        for test in sorted(categories["should_be_daily"], key=lambda t: t["file"]):
            rel_path = Path(test["file"]).relative_to(Path.cwd())
            print(f"- `{rel_path}::{test['name']}` - {test['lines']} lines")
            reasons = []
            if test["likely_integration"]:
                reasons.append("Uses external services")
            if test["lines"] > 100:
                reasons.append("Long test (>100 lines)")
            if "integration" in test["markers"]:
                reasons.append("Already marked as integration")
            if reasons:
                print(f"  - Reason: {', '.join(reasons)}")
    else:
        print("*No additional tests identified for daily execution*")

    print("\n## Summary Statistics\n")
    total_tests = sum(len(tests) for tests in categories.values())
    print(f"- Total tests analyzed: {total_tests}")
    print(f"- Already marked: {len(categories['already_marked'])}")
    print(f"- Recommended for hourly: {len(categories['should_be_hourly'])}")
    print(f"- Recommended for daily: {len(categories['should_be_daily'])}")
    print(f"- Unclassified: {len(categories['unclassified'])}")

    print("\n## How to Apply Recommendations\n")
    print("1. Review the recommendations above")
    print("2. For each test, add the appropriate marker:")
    print("   ```python")
    print("   @pytest.mark.hourly  # For critical path tests")
    print("   def test_example():")
    print("       pass")
    print("   ```")
    print("3. Run `pytest -m hourly` locally to verify")
    print("4. Update documentation if adding new test categories")


def main():
    """Main entry point."""
    root_dir = Path.cwd()
    all_tests = []

    # Find all test files
    test_patterns = ["**/test_*.py", "**/*_test.py"]
    test_files = []
    for pattern in test_patterns:
        test_files.extend(root_dir.rglob(pattern))

    # Filter out virtual environments and build directories
    test_files = [f for f in test_files if not any(part in str(f) for part in ["venv", "__pycache__", "build", ".tox"])]

    print(f"Analyzing {len(test_files)} test files...\n")

    # Analyze each file
    for test_file in test_files:
        tests = analyze_test_file(test_file)
        all_tests.extend(tests)

    # Categorize and generate recommendations
    categories = categorize_tests(all_tests)
    generate_recommendations(categories)


if __name__ == "__main__":
    main()
