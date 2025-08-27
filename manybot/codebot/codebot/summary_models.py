"""
PydanticAI models for code summarization.

These models define the structured outputs for the summarize command,
ensuring consistent and parseable results from AI operations.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class CodeComponent(BaseModel):
    """A significant code component identified in the summary."""

    name: str = Field(description="Name of the component")
    type: Literal["class", "function", "module", "interface", "constant", "variable"] = Field(
        description="Type of code component"
    )
    description: str = Field(description="Brief description of what this component does")
    file_path: str = Field(description="File path where this component is defined")
    dependencies: List[str] = Field(default_factory=list, description="Other components this depends on")
    importance: Literal["high", "medium", "low"] = Field(default="medium", description="Importance level")


class CodePattern(BaseModel):
    """Identified pattern or convention in the codebase."""

    pattern: str = Field(description="Name or type of the pattern")
    description: str = Field(description="Description of the pattern and its usage")
    examples: List[str] = Field(default_factory=list, description="Example usages of this pattern")
    frequency: Literal["common", "occasional", "rare"] = Field(
        default="common", description="How frequently this pattern appears"
    )


class SummaryResult(BaseModel):
    """Structured output from code summarization."""

    overview: str = Field(description="High-level overview of the codebase (under 500 words)")
    components: List[CodeComponent] = Field(
        default_factory=list, description="Key components identified in the codebase"
    )
    external_dependencies: List[str] = Field(default_factory=list, description="External packages/libraries used")
    patterns: List[CodePattern] = Field(default_factory=list, description="Common patterns and conventions")
    entry_points: List[str] = Field(
        default_factory=list, description="Main entry points into the code (main functions, CLI commands, etc.)"
    )
    architecture_notes: str = Field(default="", description="Notes about the overall architecture and design")

    @field_validator("overview")
    @classmethod
    def overview_not_too_long(cls, v: str) -> str:
        """Ensure overview stays concise."""
        word_count = len(v.split())
        if word_count > 500:
            raise ValueError(f"Overview must be under 500 words, got {word_count}")
        return v

    @field_validator("components")
    @classmethod
    def limit_components(cls, v: List[CodeComponent]) -> List[CodeComponent]:
        """Limit number of components to keep summary focused."""
        if len(v) > 20:
            # Keep only high and medium importance components
            filtered = [c for c in v if c.importance in ["high", "medium"]]
            if len(filtered) > 20:
                filtered = filtered[:20]
            return filtered
        return v

    def to_markdown(self) -> str:
        """Convert to readable markdown format."""
        sections = [f"# Code Summary\n\n{self.overview}"]

        if self.components:
            sections.append("\n## Key Components\n")
            # Group by importance
            high_importance = [c for c in self.components if c.importance == "high"]
            medium_importance = [c for c in self.components if c.importance == "medium"]
            low_importance = [c for c in self.components if c.importance == "low"]

            for importance, components in [
                ("High Priority", high_importance),
                ("Medium Priority", medium_importance),
                ("Low Priority", low_importance),
            ]:
                if components:
                    sections.append(f"\n### {importance}\n")
                    for c in components:
                        sections.append(f"- **{c.name}** ({c.type}): {c.description}")
                        if c.file_path:
                            sections.append(f"  - Location: `{c.file_path}`")
                        if c.dependencies:
                            deps_str = ", ".join(c.dependencies[:3])  # Limit to first 3
                            if len(c.dependencies) > 3:
                                deps_str += f" (+{len(c.dependencies) - 3} more)"
                            sections.append(f"  - Dependencies: {deps_str}")

        if self.external_dependencies:
            sections.append("\n## External Dependencies\n")
            sections.extend(f"- {d}" for d in sorted(self.external_dependencies))

        if self.patterns:
            sections.append("\n## Common Patterns\n")
            for p in self.patterns:
                sections.append(f"- **{p.pattern}** ({p.frequency}): {p.description}")
                if p.examples:
                    example_str = ", ".join(f"`{ex}`" for ex in p.examples[:2])  # Limit to first 2
                    if len(p.examples) > 2:
                        example_str += f" (+{len(p.examples) - 2} more)"
                    sections.append(f"  - Examples: {example_str}")

        if self.entry_points:
            sections.append("\n## Entry Points\n")
            sections.extend(f"- `{ep}`" for ep in self.entry_points)

        if self.architecture_notes:
            sections.append(f"\n## Architecture Notes\n\n{self.architecture_notes}")

        return "\n".join(sections)

    def to_json_summary(self) -> dict:
        """Convert to a compact JSON-friendly summary."""
        return {
            "overview_word_count": len(self.overview.split()),
            "component_count": len(self.components),
            "dependency_count": len(self.external_dependencies),
            "pattern_count": len(self.patterns),
            "entry_point_count": len(self.entry_points),
            "high_priority_components": len([c for c in self.components if c.importance == "high"]),
            "medium_priority_components": len([c for c in self.components if c.importance == "medium"]),
            "low_priority_components": len([c for c in self.components if c.importance == "low"]),
        }
