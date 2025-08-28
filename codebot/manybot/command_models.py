"""
PydanticAI models for codebot commands.

These models define the structured outputs for various codebot commands,
ensuring consistent and parseable results from AI operations.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TestCase(BaseModel):
    """A single test case to generate."""

    test_name: str = Field(description="Descriptive test function name")
    test_type: Literal["unit", "integration", "edge_case", "property", "performance"] = Field(
        description="Type of test"
    )
    imports_needed: List[str] = Field(default_factory=list, description="Import statements required")
    test_code: str = Field(description="Complete test function code")
    description: str = Field(default="", description="Brief description of what this test verifies")
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Test priority")

    @field_validator("test_name")
    @classmethod
    def valid_python_name(cls, v: str) -> str:
        """Ensure test name is valid Python and starts with test_."""
        # Clean up the name
        clean_name = v.replace(" ", "_").replace("-", "_").lower()
        clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")

        if not clean_name.startswith("test_"):
            clean_name = f"test_{clean_name}"

        return clean_name


class TestGenerationResult(BaseModel):
    """Result of test generation command."""

    test_cases: List[TestCase] = Field(description="Generated test cases")
    setup_code: Optional[str] = Field(default=None, description="Setup code for tests")
    teardown_code: Optional[str] = Field(default=None, description="Teardown code for tests")
    additional_fixtures: List[str] = Field(default_factory=list, description="Additional test fixtures")
    imports: List[str] = Field(default_factory=list, description="Common imports for all tests")
    test_framework: str = Field(default="pytest", description="Testing framework used")

    @field_validator("test_cases")
    @classmethod
    def limit_test_cases(cls, v: List[TestCase]) -> List[TestCase]:
        """Limit number of test cases to keep focused."""
        if len(v) > 25:
            # Prioritize high and medium priority tests
            prioritized = sorted(v, key=lambda t: {"high": 3, "medium": 2, "low": 1}[t.priority], reverse=True)
            return prioritized[:25]
        return v

    def to_test_file(self, target_module: str = "") -> str:
        """Convert to a complete test file."""
        sections = []

        # Import section
        all_imports = set(self.imports)
        for test in self.test_cases:
            all_imports.update(test.imports_needed)

        if "pytest" not in all_imports and self.test_framework == "pytest":
            all_imports.add("import pytest")

        if all_imports:
            sections.append("\n".join(sorted(all_imports)))

        # Target module imports
        if target_module:
            sections.append(f"\n# Import the module under test\nfrom {target_module} import *")

        # Setup code
        if self.setup_code:
            sections.append(f"\n# Setup\n{self.setup_code}")

        # Fixtures
        if self.additional_fixtures:
            sections.append("\n# Fixtures\n" + "\n\n".join(self.additional_fixtures))

        # Test cases
        if self.test_cases:
            sections.append("\n# Test Cases")
            test_code_parts = []
            for test in self.test_cases:
                test_with_docstring = test.test_code
                if test.description and '"""' not in test.test_code:
                    # Add docstring if not present
                    lines = test_with_docstring.split("\n")
                    if lines and lines[0].strip().endswith(":"):
                        lines.insert(1, f'    """{test.description}"""')
                        test_with_docstring = "\n".join(lines)
                test_code_parts.append(test_with_docstring)
            sections.append("\n\n".join(test_code_parts))

        # Teardown code
        if self.teardown_code:
            sections.append(f"\n# Teardown\n{self.teardown_code}")

        return "\n\n".join(filter(None, sections))


class CodeFix(BaseModel):
    """A specific code fix to apply."""

    file_path: str = Field(description="Path to file that needs fixing")
    fix_type: Literal["bug_fix", "style_improvement", "performance", "security", "refactor"] = Field(
        description="Type of fix being applied"
    )
    description: str = Field(description="Description of what the fix does")
    original_code: Optional[str] = Field(default=None, description="Original problematic code")
    fixed_code: str = Field(description="Fixed/improved code")
    line_range: Optional[str] = Field(default=None, description="Line range affected (e.g., '10-15')")
    reason: str = Field(description="Explanation of why this fix is needed")
    risk_level: Literal["low", "medium", "high"] = Field(default="medium", description="Risk level of this fix")


class FixResult(BaseModel):
    """Result of general fix command."""

    fixes: List[CodeFix] = Field(description="List of fixes to apply")
    summary: str = Field(description="Overall summary of fixes made")
    files_affected: List[str] = Field(default_factory=list, description="List of files that will be changed")

    def to_file_changes(self) -> List[dict]:
        """Convert fixes to file changes format."""
        file_changes = {}

        # Group fixes by file
        for fix in self.fixes:
            if fix.file_path not in file_changes:
                file_changes[fix.file_path] = {
                    "filepath": fix.file_path,
                    "content": fix.fixed_code,
                    "fixes_applied": [],
                }
            file_changes[fix.file_path]["fixes_applied"].append(
                {"type": fix.fix_type, "description": fix.description, "reason": fix.reason}
            )

        return list(file_changes.values())


class RefactorOperation(BaseModel):
    """A single refactoring operation."""

    operation_type: Literal["extract_function", "rename", "move_class", "simplify_logic", "remove_duplication"] = Field(
        description="Type of refactoring operation"
    )
    description: str = Field(description="Description of the refactoring")
    file_path: str = Field(description="File being refactored")
    impact: Literal["low", "medium", "high"] = Field(description="Impact level of this refactoring")
    reason: str = Field(description="Why this refactoring improves the code")


class RefactoringResult(BaseModel):
    """Result of refactoring command."""

    operations: List[RefactorOperation] = Field(description="List of refactoring operations performed")
    refactored_files: List[dict] = Field(description="Files with their refactored content")
    summary: str = Field(description="Summary of refactoring changes")
    quality_improvements: List[str] = Field(default_factory=list, description="Quality improvements achieved")

    def estimate_complexity_reduction(self) -> dict:
        """Estimate how much complexity was reduced."""
        high_impact = len([op for op in self.operations if op.impact == "high"])
        medium_impact = len([op for op in self.operations if op.impact == "medium"])
        low_impact = len([op for op in self.operations if op.impact == "low"])

        return {
            "high_impact_changes": high_impact,
            "medium_impact_changes": medium_impact,
            "low_impact_changes": low_impact,
            "total_operations": len(self.operations),
        }


class ImplementationStep(BaseModel):
    """A single implementation step."""

    step_name: str = Field(description="Name of this implementation step")
    file_path: str = Field(description="File to create or modify")
    code: str = Field(description="Code to implement")
    step_type: Literal["create_file", "modify_function", "add_class", "add_import", "add_test"] = Field(
        description="Type of implementation step"
    )
    dependencies: List[str] = Field(default_factory=list, description="Other steps this depends on")
    description: str = Field(description="What this step accomplishes")


class ImplementationResult(BaseModel):
    """Result of implementation command."""

    implementation_steps: List[ImplementationStep] = Field(description="Steps to implement the feature")
    files_to_create: List[str] = Field(default_factory=list, description="New files that will be created")
    files_to_modify: List[str] = Field(default_factory=list, description="Existing files that will be modified")
    summary: str = Field(description="Summary of what will be implemented")
    estimated_complexity: Literal["simple", "moderate", "complex"] = Field(description="Implementation complexity")

    def get_implementation_order(self) -> List[ImplementationStep]:
        """Get implementation steps in dependency order."""
        # Simple dependency resolution - in a real implementation,
        # you might want a more sophisticated topological sort
        ordered_steps = []
        remaining_steps = self.implementation_steps.copy()

        while remaining_steps:
            # Find steps with no unmet dependencies
            ready_steps = []
            for step in remaining_steps:
                if not step.dependencies or all(
                    dep in [s.step_name for s in ordered_steps] for dep in step.dependencies
                ):
                    ready_steps.append(step)

            if not ready_steps:
                # Circular dependency or error - just take remaining steps
                ordered_steps.extend(remaining_steps)
                break

            # Add ready steps and remove from remaining
            ordered_steps.extend(ready_steps)
            for step in ready_steps:
                remaining_steps.remove(step)

        return ordered_steps


class TestFailure(BaseModel):
    """Information about a failing test."""

    test_name: str = Field(description="Name of the failing test")
    error_type: str = Field(description="Type of error (e.g., AssertionError, ValueError)")
    error_message: str = Field(description="Error message from test failure")
    traceback: str = Field(description="Stack trace from the failure")
    suspected_cause: str = Field(description="Analysis of what likely caused the failure")
    file_path: str = Field(description="File containing the failing test")


class DebugFix(BaseModel):
    """A fix for a debugging issue."""

    target_file: str = Field(description="File that needs to be fixed")
    fix_description: str = Field(description="What the fix does")
    original_code: Optional[str] = Field(default=None, description="Original problematic code")
    fixed_code: str = Field(description="Code that fixes the issue")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this fix")
    affects_tests: bool = Field(default=False, description="Whether this fix modifies test code")


class DebugResult(BaseModel):
    """Result of debugging test failures."""

    failures_analyzed: List[TestFailure] = Field(description="Test failures that were analyzed")
    fixes_proposed: List[DebugFix] = Field(description="Fixes to resolve the failures")
    root_cause_analysis: str = Field(description="Analysis of the root cause of failures")
    confidence_level: Literal["high", "medium", "low"] = Field(
        default="medium", description="Overall confidence in the proposed fixes"
    )

    def get_implementation_fixes(self) -> List[DebugFix]:
        """Get fixes that modify implementation code (not tests)."""
        return [fix for fix in self.fixes_proposed if not fix.affects_tests]

    def get_test_fixes(self) -> List[DebugFix]:
        """Get fixes that modify test code."""
        return [fix for fix in self.fixes_proposed if fix.affects_tests]
