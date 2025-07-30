# Claude Review Tools

This directory contains specialized tools designed to assist Claude in code reviews by providing structured analysis.

## Available Tools

### type_check.py

A type annotation analyzer that helps Claude identify missing type annotations in Python code.

**Commands:**

- `analyze <file.py>` - Analyze file(s) for missing type annotations
- `mypy <file.py>` - Run mypy type checker
- `pyright <file.py>` - Run pyright type checker
- `check <file.py>` - Run all checks and provide comprehensive summary

**Features:**

- AST-based analysis to identify missing annotations
- Smart filtering to only report high-value annotations
- Structured JSON output for easy parsing
- Respects coding conventions (skips private methods, properties, etc.)

**Usage in Claude workflows:**

```bash
# Analyze a single file
.github/scripts/claude_tools/type_check.py analyze src/example.py

# Analyze multiple files
.github/scripts/claude_tools/type_check.py analyze src/*.py

# Run comprehensive check
.github/scripts/claude_tools/type_check.py check src/example.py
```
