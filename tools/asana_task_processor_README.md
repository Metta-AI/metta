# Asana Task Processor

This tool automatically processes tasks from Asana's External Projects (Inbox), analyzes the codebase to generate detailed descriptions, and moves tasks to the "Specified" section.

## Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Configure Asana Credentials

You'll need the following information from your Asana account:

- **API Token**: Generate a Personal Access Token from Asana:
  1. Go to https://app.asana.com/0/my-apps
  2. Click "Create new token"
  3. Give it a name and copy the token

- **Workspace ID**: Find your workspace ID:
  1. Go to your Asana workspace
  2. The URL will be like: `https://app.asana.com/0/WORKSPACE_ID/...`
  3. Copy the WORKSPACE_ID part

- **Project ID**: Find your External Projects (Inbox) project ID:
  1. Navigate to your External Projects (Inbox) in Asana
  2. The URL will be like: `https://app.asana.com/0/PROJECT_ID/...`
  3. Copy the PROJECT_ID part

- **Specified Section ID** (optional): Find the "Specified" section ID:
  1. Click on the "Specified" section in your project
  2. The URL will change to include the section ID
  3. Or leave empty and the script will try to find it automatically

### 3. Set Environment Variables (Optional)

You can set these as environment variables to avoid entering them each time:

```bash
export ASANA_API_TOKEN="your-api-token"
export ASANA_WORKSPACE_ID="your-workspace-id"
export ASANA_PROJECT_ID="your-project-id"
export ASANA_SPECIFIED_SECTION_ID="your-specified-section-id"  # optional
```

## Usage

Run the script:

```bash
python /workspace/tools/asana_task_processor.py
```

If environment variables are not set, the script will prompt you for the required information.

## What the Script Does

1. **Fetches Inbox Tasks**: Gets all tasks from the External Projects (Inbox) that are not in any section or explicitly in the inbox section

2. **Analyzes Codebase**: For each task:
   - Extracts keywords from the task title and description
   - Searches the codebase for relevant code references
   - Analyzes project structure
   - Suggests implementation approaches
   - Generates technical specifications

3. **Updates Tasks**:
   - Updates the task description with the detailed analysis
   - Moves the task to the "Specified" section
   - Provides links to the updated tasks

## Features

- **Smart Keyword Extraction**: Identifies technical terms from task titles and descriptions
- **Code Search**: Searches Python, YAML, JavaScript, TypeScript files for relevant code
- **Context-Aware Suggestions**: Provides different implementation approaches based on task type (API, UI, testing, etc.)
- **Project Structure Analysis**: Shows relevant directories and file counts
- **Rate Limiting**: Includes delays to respect Asana API limits

## Example Output

```
Starting Asana task processing...
Found 3 tasks in inbox

============================================================
Processing task 1/3: Add unit tests for authentication module
============================================================

Analyzing codebase for task: Add unit tests for authentication module

Keywords identified: unit, tests, authentication, module

Found references to 'authentication':
- metta/auth/manager.py:45: class AuthenticationManager:
- tests/auth/test_basic.py:12: def test_authentication_flow():

Updating task description and moving to 'Specified'...
✓ Successfully processed task: Add unit tests for authentication module
  Task URL: https://app.asana.com/0/123456789/987654321
```

## Troubleshooting

- **"Could not find 'Specified' section"**: Make sure you have a section named "Specified" in your project, or set the `ASANA_SPECIFIED_SECTION_ID` environment variable
- **API Errors**: Check that your API token has the necessary permissions for the workspace and project
- **No tasks found**: Ensure tasks are in the inbox (not assigned to any section) in your External Projects
