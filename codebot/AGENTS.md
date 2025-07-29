# Command Implementation Guide

This document provides guidance for implementing new commands in the Codebot framework.

## Command Structure

Each command is a focused operation that:
1. Gathers relevant context automatically
2. Executes via the appropriate LLM mode
3. Returns structured file changes

## Implementation Template

```python
from dataclasses import dataclass
from typing import List
from codebot.core import Command, OutputSchema

@dataclass
class MyCommand(Command):
    def __init__(self):
        super().__init__(
            name="mycommand",
            description="Brief description of what this command does",
            prompt_template=self.load_prompt("mycommand.md"),
            default_paths=["relevant/", "paths/"],
            output_schema=OutputSchema(
                file_patterns=["output_*.py"],
                required_fields=["file_changes", "summary"]
            )
        )
```

## Prompt Design

Prompts should be:
- Clear about the task and expected output
- Include specific guidelines for the domain
- Request structured output (JSON for one-shot/persistent, natural for interactive)

Example prompt structure:
```markdown
You are an expert in {domain}. Your task is to {specific_task}.

Guidelines:
- {guideline_1}
- {guideline_2}

Context will include:
- Git diff showing recent changes
- Clipboard content with user intent
- Relevant project files

{mode_specific_instructions}
```

## Context Configuration

### Git Paths
Across all commands: Include diff and full versions of all changed files

### User Paths

`codebot test mettagrid/src/curriculum` would automatically include the curriculum fileset in addtiion to the default.

### Agent Paths

`codebot test` could potentially automatically include tests/ directories.

### Token Management

Commands should specify reasonable token budgets:
- Simple commands: 10-20k tokens
- Complex analysis: 30-50k tokens
- Full implementation: 50-100k tokens

## Output Handling

### File Change Patterns

Define clear patterns for output files:
```python
output_schema=OutputSchema(
    file_patterns=["test_*.py", "*_test.py"],  # For test command
    file_patterns=["*.review.md"],              # For review command
    file_patterns=["src/**/*.py"],              # For implement command
)
```

### Validation

Implement output validation:
```python
def validate_output(self, output: CommandOutput) -> bool:
    # Check required fields exist
    if not all(field in output.metadata for field in self.output_schema.required_fields):
        return False

    # Validate file paths match patterns
    for change in output.file_changes:
        if not self.matches_pattern(change.filepath, self.output_schema.file_patterns):
            return False

    return True
```

## Testing Commands

Every command should have tests covering:
1. Context gathering
2. Prompt generation
3. Output parsing
4. File change application
5. Error handling

Example test:
```python
def test_command_execution():
    command = MyCommand()
    context = mock_context(git_diff="...", clipboard="...")

    output = command.execute(context)

    assert len(output.file_changes) > 0
    assert output.summary != ""
    assert all(validate_file_change(fc) for fc in output.file_changes)
```

## Common Patterns

### Multi-file Output
```python
# Generate multiple related files
output = CommandOutput(
    file_changes=[
        FileChange(filepath="src/feature.py", content="...", operation="write"),
        FileChange(filepath="tests/test_feature.py", content="...", operation="write"),
        FileChange(filepath="docs/feature.md", content="...", operation="write"),
    ]
)
```

### Conditional Output
```python
# Include files based on analysis
if needs_config:
    output.file_changes.append(
        FileChange(filepath="config.yaml", content="...", operation="write")
    )
```

### Progress Reporting
```python
# For long-running commands in persistent mode
output.metadata["progress"] = {
    "phase": "analyzing",
    "files_processed": 10,
    "total_files": 50
}
```
