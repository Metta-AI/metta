# Asana External Projects Implementation

## Overview

This document describes the implementation of tools for fetching and implementing tasks from Asana's External Projects. The implementation includes:

1. **Asana External Tasks Tool** - A tool to fetch tasks from Asana projects
2. **Demo Task Implementation** - A configuration validator for mettagrid environments
3. **Comprehensive Test Suite** - Tests for both tools

## Components

### 1. Asana External Tasks Tool (`tools/asana_external_tasks.py`)

A command-line tool that:
- Connects to the Asana API using a personal access token
- Finds and lists projects in a workspace
- Specifically searches for "External Projects"
- Fetches and displays tasks with full details
- Allows interactive task selection
- Creates a starter implementation file for the selected task

**Usage:**
```bash
# Set environment variables
export ASANA_API_TOKEN="your_token_here"
export ASANA_WORKSPACE_ID="workspace_id"  # Optional
export ASANA_PROJECT_ID="project_id"      # Optional

# List all projects
uv run python tools/asana_external_tasks.py --list-projects

# Fetch tasks from External Projects
uv run python tools/asana_external_tasks.py

# Show completed tasks too
uv run python tools/asana_external_tasks.py --show-completed
```

### 2. Demo Task Implementation (`tools/demo_asana_task_implementation.py`)

As a demonstration, we implemented a task that would typically come from External Projects:
**"Add configuration validation for mettagrid environments"**

The validator:
- Validates YAML configuration files for mettagrid environments
- Checks required fields (defaults, game)
- Validates game configuration parameters
- Checks object counts and dimensions
- Provides helpful error messages and warnings
- Can validate individual files or all configs

**Usage:**
```bash
# Demo mode (shows task details and validates example)
uv run python tools/demo_asana_task_implementation.py

# Validate a specific file
uv run python tools/demo_asana_task_implementation.py configs/env/mettagrid/arena/tag.yaml

# Validate all mettagrid configs
uv run python tools/demo_asana_task_implementation.py --all
```

### 3. Test Suite

Comprehensive tests ensure the tools work correctly:

- **`tests/tools/test_asana_external_tasks.py`** - Tests for the Asana API client
  - API request mocking
  - Workspace and project fetching
  - Task retrieval and display
  - Interactive task selection
  - Main function integration

- **`tests/tools/test_demo_asana_task_implementation.py`** - Tests for the validator
  - Required field validation
  - Type checking for parameters
  - Range validation
  - Warning generation for high values
  - File I/O and error handling

## Configuration

Create a `.env.asana` file (see `.env.asana.example`):
```
ASANA_API_TOKEN=your_asana_api_token_here
ASANA_WORKSPACE_ID=optional_workspace_id
ASANA_PROJECT_ID=optional_project_id
```

## Testing

Run all tests:
```bash
uv run pytest tests/tools/test_asana_external_tasks.py tests/tools/test_demo_asana_task_implementation.py -v
```

## Results

When run on the actual mettagrid configurations, the validator found:
- ✅ 70 valid configuration files
- ❌ 99 configuration files with errors (mostly template/partial files)
- ⚠️ 1 configuration with warnings (very large map dimensions)

Common issues found:
- Missing required fields in template files
- Hydra variable references (${...}) in partial configs
- Missing 'type' field in map root configurations

This demonstrates that the validator is working correctly and would be valuable as part of the test suite and CI/CD pipeline.