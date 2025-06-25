# Asana Task Processor - Implementation Summary

## Overview

I've created a comprehensive Asana task processing system that:
1. Fetches tasks from Asana's External Projects (Inbox)
2. Analyzes the codebase to generate detailed task descriptions
3. Updates task descriptions and moves tasks to the "Specified" section

## Files Created

### 1. `/workspace/tools/asana_task_processor.py`
The main production script that:
- Connects to Asana API using provided credentials
- Fetches tasks from the inbox (tasks without sections)
- Analyzes the codebase for each task:
  - Extracts keywords from task title and description
  - Searches for relevant code references
  - Analyzes project structure
  - Suggests implementation approaches
  - Generates technical specifications
- Updates task descriptions with the analysis
- Moves tasks to the "Specified" section

### 2. `/workspace/tools/asana_task_processor_README.md`
Complete documentation including:
- Setup instructions
- How to obtain Asana API credentials
- Environment variable configuration
- Usage examples
- Troubleshooting guide

### 3. `/workspace/tools/asana_task_processor_demo.py`
A demonstration script that shows how the processor works without requiring actual Asana credentials. It simulates processing three example tasks.

## Key Features

1. **Smart Keyword Extraction**: Automatically identifies technical terms from task descriptions
2. **Codebase Analysis**: Searches Python, YAML, JavaScript, and TypeScript files for relevant code
3. **Context-Aware Suggestions**: Provides different implementation approaches based on task type
4. **Project Structure Analysis**: Shows relevant directories and file counts
5. **Rate Limiting**: Includes delays to respect Asana API limits

## Usage

### Production Mode
```bash
# Set environment variables
export ASANA_API_TOKEN="your-api-token"
export ASANA_WORKSPACE_ID="your-workspace-id"
export ASANA_PROJECT_ID="your-project-id"
export ASANA_SPECIFIED_SECTION_ID="your-section-id"  # optional

# Run the processor
python /workspace/tools/asana_task_processor.py
```

### Demo Mode
```bash
# Run the demo to see how it works
python /workspace/tools/asana_task_processor_demo.py
```

## How It Works

1. **Task Fetching**: Connects to Asana and retrieves all tasks from the inbox
2. **Analysis**: For each task:
   - Extracts keywords (filtering out common words)
   - Searches codebase using grep for relevant files
   - Analyzes project structure
   - Generates implementation suggestions based on task type
3. **Update & Move**: Updates task description with the analysis and moves to "Specified" section

## Example Output

The processor generates detailed task descriptions like:

```markdown
## Task Analysis

**Task Title**: Implement caching for API endpoints

### Codebase Analysis
**Keywords identified**: caching, api, endpoints, redis

#### Found references to 'api':
- app_backend/routes/dashboard_routes.py:23: @router.get("/api/dashboard")

### Project Structure Analysis
**Key Components**:
- app_backend/: Backend API implementation
  - Python files: 10

### Implementation Approach
1. API Development: Implement in app_backend/routes/
2. Follow FastAPI patterns used in existing routes

### Technical Specifications
**Development Environment**:
- Python 3.11+ (primary language)
- FastAPI for API development
```

## Next Steps

To use this in production:
1. Obtain your Asana API credentials
2. Set up environment variables
3. Run the processor to automatically update all inbox tasks

The system is ready to process tasks from Asana's External Projects (Inbox) and move them to the "Specified" section with detailed technical descriptions based on codebase analysis.
