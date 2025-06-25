#!/usr/bin/env python3
"""
Asana Task Processor
-------------------
This script:
1. Fetches tasks from Asana's External Projects (Inbox)
2. Analyzes the codebase to generate detailed task descriptions
3. Updates task descriptions and moves tasks to "Specified" section
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# Asana API Configuration
def get_asana_config():
    """Get Asana configuration from environment or prompt user"""
    config = {
        "api_token": os.getenv("ASANA_API_TOKEN", ""),
        "workspace_id": os.getenv("ASANA_WORKSPACE_ID", ""),
        "project_id": os.getenv("ASANA_PROJECT_ID", ""),
        "specified_section_id": os.getenv("ASANA_SPECIFIED_SECTION_ID", ""),
    }

    # If not set in environment, prompt for them
    if not config["api_token"]:
        print("Please enter your Asana API Token:")
        config["api_token"] = input().strip()

    if not config["workspace_id"]:
        print("Please enter your Asana Workspace ID:")
        config["workspace_id"] = input().strip()

    if not config["project_id"]:
        print("Please enter your Asana Project ID (External Projects/Inbox):")
        config["project_id"] = input().strip()

    return config


# Get configuration
ASANA_CONFIG = get_asana_config()


class AsanaTaskProcessor:
    def __init__(self):
        self.base_url = "https://app.asana.com/api/1.0"
        self.headers = {"Authorization": f"Bearer {ASANA_CONFIG['api_token']}", "Content-Type": "application/json"}
        self.workspace_root = Path("/workspace")

    def get_project_sections(self) -> List[Dict[str, Any]]:
        """Get all sections in the project"""
        url = f"{self.base_url}/projects/{ASANA_CONFIG['project_id']}/sections"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            print(f"Error fetching sections: {response.status_code} - {response.text}")
            return []

        return response.json()["data"]

    def find_specified_section(self) -> Optional[str]:
        """Find the 'Specified' section ID"""
        sections = self.get_project_sections()

        for section in sections:
            if "specified" in section.get("name", "").lower():
                return section["gid"]

        print("Warning: Could not find 'Specified' section")
        return None

    def get_inbox_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks from the inbox (tasks not in any section or in inbox section)"""
        url = f"{self.base_url}/projects/{ASANA_CONFIG['project_id']}/tasks"
        params = {
            "opt_fields": "name,notes,gid,memberships.section,memberships.section.name,created_at,tags,custom_fields"
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching tasks: {response.status_code} - {response.text}")
            return []

        tasks = response.json()["data"]

        # Filter for inbox tasks (no section or inbox section)
        inbox_tasks = []
        for task in tasks:
            memberships = task.get("memberships", [])

            # Check if task is in inbox (no section or explicitly in inbox section)
            is_inbox = True
            for membership in memberships:
                section = membership.get("section")
                if section:
                    section_name = section.get("name", "").lower()
                    if "inbox" not in section_name and section_name != "":
                        is_inbox = False
                        break

            if is_inbox:
                inbox_tasks.append(task)

        return inbox_tasks

    def analyze_codebase_for_task(self, task_name: str, task_notes: str) -> str:
        """Analyze the codebase to generate a detailed task description"""
        print(f"\nAnalyzing codebase for task: {task_name}")

        analysis = []
        analysis.append("## Task Analysis\n")
        analysis.append(f"**Task Title**: {task_name}\n")

        if task_notes:
            analysis.append(f"**Original Description**: {task_notes}\n")

        # Extract keywords from task name and notes
        keywords = self._extract_keywords(task_name + " " + (task_notes or ""))

        analysis.append("\n### Codebase Analysis\n")

        # Search for relevant files and patterns
        if keywords:
            analysis.append(f"**Keywords identified**: {', '.join(keywords)}\n")

            # Search for each keyword in the codebase
            for keyword in keywords:
                findings = self._search_codebase(keyword)
                if findings:
                    analysis.append(f"\n#### Found references to '{keyword}':\n")
                    for finding in findings[:10]:  # Limit to 10 findings per keyword
                        analysis.append(f"- `{finding['file']}:{finding['line']}`: {finding['context']}\n")

        # Analyze project structure
        structure_analysis = self._analyze_project_structure()
        analysis.append("\n### Project Structure Analysis\n")
        analysis.append(structure_analysis)

        # Suggest implementation approach
        analysis.append("\n### Implementation Approach\n")
        implementation = self._suggest_implementation(task_name, keywords)
        analysis.append(implementation)

        # Add technical specifications
        analysis.append("\n### Technical Specifications\n")
        specs = self._generate_tech_specs(task_name, keywords)
        analysis.append(specs)

        return "".join(analysis)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words and extract technical terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "need",
            "want",
            "add",
            "fix",
            "update",
            "change",
            "modify",
            "improve",
            "implement",
            "create",
            "make",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())

        # Filter out stop words and duplicates
        keywords = []
        seen = set()
        for word in words:
            if word not in stop_words and word not in seen:
                keywords.append(word)
                seen.add(word)

        return keywords[:8]  # Limit to 8 keywords

    def _search_codebase(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for keyword in codebase"""
        findings = []

        try:
            # Use grep to search for the keyword
            cmd = [
                "grep",
                "-rn",
                "--include=*.py",
                "--include=*.yaml",
                "--include=*.yml",
                "--include=*.js",
                "--include=*.ts",
                "--include=*.jsx",
                "--include=*.tsx",
                "-i",
                keyword,
                str(self.workspace_root),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[:20]  # Limit results

                for line in lines:
                    if ":" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            file_path = parts[0].replace(str(self.workspace_root) + "/", "")
                            line_num = parts[1]
                            context = parts[2].strip()[:100]  # Limit context length

                            findings.append({"file": file_path, "line": line_num, "context": context})
        except Exception as e:
            print(f"Error searching for {keyword}: {e}")

        return findings

    def _analyze_project_structure(self) -> str:
        """Analyze the project structure"""
        analysis = []

        # Key directories
        key_dirs = {
            "metta": "Core library and main application code",
            "mettagrid": "Grid-based environment implementation",
            "mettascope": "Visualization and analysis tools",
            "observatory": "Monitoring and observability components",
            "app_backend": "Backend API implementation",
            "configs": "Configuration files for various components",
            "tests": "Test suites for all components",
            "tools": "Utility scripts and tools",
        }

        analysis.append("**Key Components**:\n")
        for dir_name, description in key_dirs.items():
            dir_path = self.workspace_root / dir_name
            if dir_path.exists():
                analysis.append(f"- **{dir_name}/**: {description}\n")

                # Count files
                py_files = len(list(dir_path.rglob("*.py")))
                if py_files > 0:
                    analysis.append(f"  - Python files: {py_files}\n")

        return "".join(analysis)

    def _suggest_implementation(self, task_name: str, keywords: List[str]) -> str:
        """Suggest implementation approach based on task and keywords"""
        suggestions = []

        # Analyze task type
        task_lower = task_name.lower()

        if any(word in task_lower for word in ["test", "testing", "unit", "integration"]):
            suggestions.append("1. **Testing Focus**: Add comprehensive test coverage\n")
            suggestions.append("2. Use pytest framework (already configured in the project)\n")
            suggestions.append("3. Follow existing test patterns in `tests/` directory\n")

        elif any(word in task_lower for word in ["api", "endpoint", "route"]):
            suggestions.append("1. **API Development**: Implement in `app_backend/routes/`\n")
            suggestions.append("2. Follow FastAPI patterns used in existing routes\n")
            suggestions.append("3. Add corresponding tests in `tests/app/`\n")

        elif any(word in task_lower for word in ["ui", "frontend", "interface"]):
            suggestions.append("1. **Frontend Development**: Work with React/TypeScript components\n")
            suggestions.append("2. Check `observatory/` and `mettamap/` for existing UI patterns\n")
            suggestions.append("3. Ensure responsive design and accessibility\n")

        elif any(word in task_lower for word in ["agent", "ai", "model"]):
            suggestions.append("1. **Agent/AI Development**: Focus on `metta/agent/` components\n")
            suggestions.append("2. Review existing agent implementations and patterns\n")
            suggestions.append("3. Consider performance and scalability implications\n")

        else:
            suggestions.append("1. **General Implementation**: Review related components\n")
            suggestions.append("2. Follow project coding standards and patterns\n")
            suggestions.append("3. Ensure backward compatibility\n")

        # Add common suggestions
        suggestions.append("\n**Common Steps**:\n")
        suggestions.append("- Review existing code patterns in related modules\n")
        suggestions.append("- Add appropriate logging and error handling\n")
        suggestions.append("- Update documentation as needed\n")
        suggestions.append("- Include unit tests for new functionality\n")

        return "".join(suggestions)

    def _generate_tech_specs(self, task_name: str, keywords: List[str]) -> str:
        """Generate technical specifications"""
        specs = []

        specs.append("**Development Environment**:\n")
        specs.append("- Python 3.11+ (primary language)\n")
        specs.append("- TypeScript/React (frontend components)\n")
        specs.append("- Docker for containerization\n")
        specs.append("- pytest for testing\n")

        specs.append("\n**Key Dependencies**:\n")
        # Check for relevant dependencies based on keywords
        if any(k in ["api", "backend", "server"] for k in keywords):
            specs.append("- FastAPI for API development\n")
            specs.append("- SQLAlchemy for database operations\n")

        if any(k in ["ml", "model", "agent", "ai"] for k in keywords):
            specs.append("- PyTorch for machine learning\n")
            specs.append("- NumPy for numerical operations\n")

        specs.append("\n**Code Standards**:\n")
        specs.append("- Follow PEP 8 for Python code\n")
        specs.append("- Use type hints for better code clarity\n")
        specs.append("- Maintain test coverage above 80%\n")
        specs.append("- Document all public APIs\n")

        return "".join(specs)

    def update_task(self, task_gid: str, new_description: str, section_gid: Optional[str] = None):
        """Update task description and optionally move to a new section"""
        url = f"{self.base_url}/tasks/{task_gid}"

        data = {"data": {"notes": new_description}}

        response = requests.put(url, headers=self.headers, json=data)

        if response.status_code != 200:
            print(f"Error updating task: {response.status_code} - {response.text}")
            return False

        # Move to specified section if provided
        if section_gid:
            self.move_task_to_section(task_gid, section_gid)

        return True

    def move_task_to_section(self, task_gid: str, section_gid: str):
        """Move task to a specific section"""
        url = f"{self.base_url}/sections/{section_gid}/addTask"

        data = {"data": {"task": task_gid}}

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code != 200:
            print(f"Error moving task to section: {response.status_code} - {response.text}")
            return False

        return True

    def process_tasks(self):
        """Main process to handle all inbox tasks"""
        print("Starting Asana task processing...")

        # Find specified section
        specified_section = ASANA_CONFIG["specified_section_id"]
        if not specified_section:
            specified_section = self.find_specified_section()
            if not specified_section:
                print("Error: Could not find 'Specified' section. Please set ASANA_SPECIFIED_SECTION_ID")
                return

        # Get inbox tasks
        inbox_tasks = self.get_inbox_tasks()

        if not inbox_tasks:
            print("No tasks found in inbox")
            return

        print(f"Found {len(inbox_tasks)} tasks in inbox")

        # Process each task
        for i, task in enumerate(inbox_tasks, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing task {i}/{len(inbox_tasks)}: {task['name']}")
            print(f"{'=' * 60}")

            # Analyze codebase and generate description
            detailed_description = self.analyze_codebase_for_task(task["name"], task.get("notes", ""))

            # Update task
            print("\nUpdating task description and moving to 'Specified'...")

            if self.update_task(task["gid"], detailed_description, specified_section):
                print(f"✓ Successfully processed task: {task['name']}")
                print(f"  Task URL: https://app.asana.com/0/{ASANA_CONFIG['project_id']}/{task['gid']}")
            else:
                print(f"✗ Failed to process task: {task['name']}")

            # Rate limiting
            time.sleep(1)

        print(f"\nCompleted processing {len(inbox_tasks)} tasks")


def main():
    """Main entry point"""
    processor = AsanaTaskProcessor()

    try:
        processor.process_tasks()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
