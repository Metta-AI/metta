# Codebot: AI-Powered Code Assistant Framework


## Usage Examples [AS A VISION; NOT YET IMPLEMENTED]

```bash
# Write tests for a specific change (uses `git diff main` and `tests/` as default context)
codebot test "Write tests for the new UserAuth class"

# Fix ruff issues (automatically pipes in ruff check output as context, along wit git diff main)
codebot ruff "Fix my linting issues"

# Run a workflow
codebot workflow test-debug "Add authentication to the API endpoints" -p src/api

# List available bots
codebot bots

# Reset --soft to last manual commit
codebot reset-soft

# See all bot changes since last manual commit
codebot diff
```

## Vision

Codebot is a framework for building AI-powered code assistants that amplify human engineering capabilities. Built on PydanticAI, it creates well-defined handoff contracts between components while adapting to the evolving capabilities of language models.

### Core Principles

1. **Human Empowerment First**: Maximize the productivity and capabilities of our engineers, not replace them with autonomous agents

2. **Best Context, Always**: Provide the most relevant context possible while adapting expectations based on model capabilities as they improve

3. **Clear Handoff Contracts**: Use Pydantic's input/output specifications to enable reliable component composition and graph structures

4. **Start Simple, Focus on Quality**: Build a small number of highly valuable bots before scaling - 4 excellent bots are better than 20 mediocre ones

5. **Experiment and Learn**: Try multiple approaches (full files vs diffs), measure failure modes, and invest based on real-world performance


## Prompt Management System

Centralized markdown-based prompts that can be used by both Codebot and Claude Code subagents:

```python
from pathlib import Path
from typing import Dict, Optional
import re

class PromptManager:
    """Manage prompts stored as markdown files"""

    def __init__(self, prompts_dir: Path = Path("prompts")):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all .md files from prompts directory"""
        for md_file in self.prompts_dir.glob("*.md"):
            name = md_file.stem  # filename without .md
            self._cache[name] = md_file.read_text()

    def get(self, name: str, **variables) -> str:
        """Get a prompt by name and optionally substitute variables"""
        if name not in self._cache:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self._cache.keys())}")

        prompt = self._cache[name]

        # Simple variable substitution for {variable_name}
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt

    def get_raw(self, name: str) -> str:
        """Get raw prompt without any substitution (for Claude Code)"""
        if name not in self._cache:
            raise KeyError(f"Prompt '{name}' not found")
        return self._cache[name]

    def list_prompts(self) -> Dict[str, str]:
        """List all available prompts with their first line as description"""
        prompts = {}
        for name, content in self._cache.items():
            # First line as description
            first_line = content.split('\n')[0].strip('#').strip()
            prompts[name] = first_line
        return prompts

# Global instance
prompts = PromptManager()
```

### Example Prompt Files

`prompts/tester.md`:
```markdown
# Test Writer - Comprehensive Unit Test Generation

You are an expert test writer focused on creating comprehensive, maintainable test suites.

## Core Responsibilities
- Write tests that verify behavior, not implementation
- Cover edge cases and error conditions
- Follow existing test patterns in the codebase
- Use appropriate test fixtures and mocking

## Guidelines
1. **Test Structure**
   - Use descriptive test names that explain what is being tested
   - Group related tests in classes
   - Follow Arrange-Act-Assert pattern

2. **Coverage Focus**
   - Happy path scenarios
   - Error conditions and exceptions
   - Boundary conditions
   - Integration points

3. **Framework**
   - Use pytest as the testing framework
   - Leverage pytest fixtures for setup/teardown
   - Use appropriate markers (@pytest.mark.asyncio, etc.)

## Context Usage
When provided with code context, analyze:
- Public APIs that need testing
- Dependencies that should be mocked
- Existing test patterns to follow
- Edge cases specific to the implementation
```

`prompts/debugger.md`:
```markdown
# Debugger - Test Failure Analysis and Resolution

You are an expert debugger specialized in fixing failing tests and resolving errors.

## Core Responsibilities
- Analyze test failures to identify root causes
- Fix bugs while maintaining existing functionality
- Ensure fixes don't introduce new issues
- Improve code quality while debugging

## Debugging Process
1. **Analyze the Error**
   - Read error messages and stack traces carefully
   - Identify the exact point of failure
   - Understand what the test expects vs what happens

2. **Investigate Root Cause**
   - Check recent changes that might have caused the issue
   - Verify assumptions about data and state
   - Look for race conditions or timing issues

3. **Implement Fix**
   - Make minimal changes to fix the issue
   - Preserve all existing functionality
   - Add defensive programming where appropriate

## Common Patterns
- Import errors: Check module paths and dependencies
- Assertion failures: Verify test expectations are correct
- Type errors: Ensure proper type handling and conversions
- Async issues: Check proper await usage and event loop handling
```

`prompts/reviewer.md`:
```markdown
# Code Reviewer - Thorough Code Quality Analysis

You are a senior engineer performing code reviews with focus on quality, security, and maintainability.

## Review Priorities
1. **Correctness**
   - Logic errors and bugs
   - Edge case handling
   - Error handling completeness

2. **Security**
   - Input validation
   - Authentication/authorization issues
   - Data exposure risks
   - Injection vulnerabilities

3. **Performance**
   - Algorithmic complexity
   - Database query efficiency
   - Memory usage patterns
   - Caching opportunities

4. **Maintainability**
   - Code clarity and readability
   - Appropriate abstractions
   - Documentation completeness
   - Test coverage

## Review Output Format
Organize feedback by severity:
- **Critical**: Must fix before merge (bugs, security issues)
- **Important**: Should address (performance, maintainability)
- **Suggestion**: Consider improving (style, minor optimizations)
- **Praise**: Highlight good practices

Always provide specific examples and suggest improvements.
```

`prompts/summarizer.md`:
```markdown
# Code Summarizer - Technical Documentation Generator

You are a technical documentation expert creating clear, concise summaries of codebases.

## Summarization Priorities
1. **APIs and Interfaces**
   - Public methods and their signatures
   - Input/output specifications
   - Usage examples

2. **Data Models**
   - Core data structures
   - Database schemas
   - State management patterns

3. **Architecture**
   - High-level system design
   - Component relationships
   - Key design decisions

4. **Integration Points**
   - External dependencies
   - API endpoints
   - Event handlers

## Compression Strategy
When given a token limit:
- Preserve all public APIs
- Keep essential type definitions
- Summarize implementation details
- Remove redundant documentation
- Compress verbose descriptions

Output should be scannable and useful for engineers who need to:
- Understand how to use the code
- Extend functionality
- Debug issues
- Review architecture decisions
```

## Simple Bot Contracts

Clear input/output specifications for single-request bots:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# Universal Input Contract
class BotInput(BaseModel):
    """Standard input for all single-request bots"""
    task_description: str = Field(description="The user's request - what they want the bot to do")
    clipboard_content: Optional[str] = Field(description="User's clipboard if relevant")
    user_context: str = Field(description="Context from user-specified paths")
    bot_context: str = Field(description="Context from bot-specific default paths")
    bot_prompt: str = Field(description="Bot-specific system prompt")

# Universal Output Contract
class FileChange(BaseModel):
    """A single file change"""
    filepath: str
    content: str
    operation: str = Field(default="write", description="write|delete")

class BotOutput(BaseModel):
    """Standard output for all single-request bots"""
    file_changes: List[FileChange]
    explanation: str = Field(description="What was done and why")

# Base contract for all simple bots
class SimpleBot(ABC):
    """Contract for single LLM request bots"""

    def __init__(self, name: str, prompt_name: Optional[str] = None):
        self.name = name
        self.prompt_name = prompt_name or name  # Default to bot name
        self.bot_prompt = prompts.get(self.prompt_name)
        self.context_provider = ContextProvider(name)
        self.agent = Agent(
            model="anthropic:claude-3-opus-20240229",
            output_type=BotOutput,
            system_prompt=self.bot_prompt
        )

    async def execute(
        self,
        task: str,
        clipboard: Optional[str] = None,
        context_spec: Optional[ContextSpec] = None
    ) -> BotOutput:
        """Execute bot with standard inputs producing standard outputs"""
        # Get split contexts
        user_context, bot_context = await self.context_provider.get_contexts(context_spec)

        # Format prompt with clear sections
        prompt = f"""
Task: {task}

{f"Clipboard Content:\n{clipboard}" if clipboard else ""}

User-Specified Context:
{user_context}

Bot Default Context:
{bot_context}
        """

        # Execute
        result = await self.agent.run(prompt)
        return result.output
```

class ContextProvider:
    """White-list only context provider with automatic defaults"""

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        # Pre-specified defaults per bot type
        self.bot_defaults = {
            "tester": ["tests/", "conftest.py"],
            "debugger": ["tests/", ".github/workflows/"],
            "summarizer": ["README.md", "docs/"],
            "reviewer": [".github/PULL_REQUEST_TEMPLATE.md"]
        }

    def _parse_changed_files_from_diff(self, diff: str) -> List[str]:
        """Extract list of changed files from git diff output"""
        changed_files = []
        for line in diff.split('\n'):
            if line.startswith('diff --git'):
                # Extract filename from diff --git a/path/to/file b/path/to/file
                parts = line.split()
                if len(parts) >= 3:
                    # Remove 'a/' prefix
                    filepath = parts[2][2:] if parts[2].startswith('a/') else parts[2]
                    changed_files.append(filepath)
        return list(set(changed_files))  # Remove duplicates

    async def get_contexts(self, user_spec: Optional[ContextSpec] = None) -> tuple[str, str]:
        """
        Build split contexts:
        - user_context: Git diff + full changed files + user-specified paths
        - bot_context: Bot-specific default paths

        Returns:
            (user_context, bot_context) tuple
        """
        user_parts = []
        bot_parts = []

        # 1. Git diff goes in user context
        diff = subprocess.run(
            ["git", "diff", "main"],
            capture_output=True,
            text=True
        ).stdout

        if diff:
            user_parts.append("=== CURRENT CHANGES (git diff main) ===")
            user_parts.append(diff)

            # Extract changed files and include their full content
            changed_files = self._parse_changed_files_from_diff(diff)
            if changed_files:
                # Get full content of changed files using codeclip
                changed_content = self._get_files_content(changed_files)
                if changed_content:
                    user_parts.append("=== FULL CONTENT OF CHANGED FILES ===")
                    user_parts.append(changed_content)

        # 2. User-specified paths go in user context
        if user_spec and user_spec.include_paths:
            user_files = self._get_files_content(user_spec.include_paths)
            if user_files:
                user_parts.append("=== USER SPECIFIED FILES ===")
                user_parts.append(user_files)

        # 3. Bot defaults go in bot context
        if defaults := self.bot_defaults.get(self.bot_name):
            bot_files = self._get_files_content(defaults)
            if bot_files:
                bot_parts.append(f"=== {self.bot_name.upper()} DEFAULT FILES ===")
                bot_parts.append(bot_files)

        user_context = "\n\n".join(user_parts) if user_parts else "No user context provided"
        bot_context = "\n\n".join(bot_parts) if bot_parts else "No bot-specific context"

        return user_context, bot_context

    def _get_files_content(self, paths: List[str]) -> str:
        """Get content using codeclip for specified paths"""
        if not paths:
            return ""

        # Filter out non-existent files
        existing_paths = [p for p in paths if Path(p).exists()]
        if not existing_paths:
            return ""

        cmd = ["metta", "clip", "-s"] + existing_paths
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else ""

## Example Bot Implementations

### Test Writer Bot
```python
class TesterBot(SimpleBot):
    def __init__(self):
        super().__init__(name="tester")

# Usage
bot = TesterBot()
output = await bot.execute(
    task="Write tests for the new UserAuth class",
    context_spec=ContextSpec(include_paths=["src/auth.py"])
)
```

### Debugger Bot
```python
class DebuggerBot(SimpleBot):
    def __init__(self):
        super().__init__(name="debugger")
```

### Code Reviewer Bot
```python
class ReviewerBot(SimpleBot):
    def __init__(self):
        super().__init__(name="reviewer")
```

## Claude Code Integration

The prompt system is designed to be used by Claude Code subagents as well:

```python
class ClaudeCodePromptExporter:
    """Export prompts for Claude Code configuration"""

    def __init__(self, prompts_dir: Path = Path("prompts")):
        self.prompts = PromptManager(prompts_dir)

    def export_for_claude(self, output_dir: Path):
        """Export prompts in format suitable for Claude Code"""
        output_dir.mkdir(exist_ok=True)

        # Create a manifest file
        manifest = {
            "prompts": []
        }

        for name in self.prompts._cache:
            # Copy prompt file
            content = self.prompts.get_raw(name)
            output_file = output_dir / f"{name}.md"
            output_file.write_text(content)

            # Add to manifest
            manifest["prompts"].append({
                "name": name,
                "file": f"{name}.md",
                "description": content.split('\n')[0].strip('#').strip()
            })

        # Write manifest
        import json
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    def get_claude_config_snippet(self) -> str:
        """Generate configuration snippet for Claude Code"""
        available = self.prompts.list_prompts()

        config = """
# Claude Code Subagent Configuration

## Available Prompts
"""
        for name, description in available.items():
            config += f"\n### {name}\n{description}\n"
            config += f"Path: `prompts/{name}.md`\n"

        return config
```

## Git Integration Layer

Automatic commit management around bot operations:

```python
class GitManager:
    """Manages git operations around bot changes"""

    def __init__(self):
        self.manual_commit_marker = "MANUAL:"
        self.bot_commit_marker = "BOT:"

    def get_last_manual_commit(self) -> Optional[str]:
        """Find the last commit made manually by a human"""
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H %s", "-n", "50"],
            capture_output=True,
            text=True
        )

        for line in result.stdout.splitlines():
            commit_hash, message = line.split(" ", 1)
            if not message.startswith(self.bot_commit_marker):
                return commit_hash

        return None

    def create_bot_checkpoint(self, bot_name: str, task: str) -> str:
        """Create automatic commit before bot changes"""
        # Stage any existing changes
        subprocess.run(["git", "add", "-A"])

        # Create checkpoint commit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        message = f"{self.bot_commit_marker} [{bot_name}] Checkpoint before: {task[:50]}"

        subprocess.run(["git", "commit", "-m", message, "--allow-empty"])

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def reset_to_last_manual(self):
        """Soft reset to last manual commit, preserving changes"""
        last_manual = self.get_last_manual_commit()
        if last_manual:
            subprocess.run(["git", "reset", "--soft", last_manual])
            print(f"Reset to manual commit: {last_manual}")

    def diff_since_last_manual(self):
        """Show diff since last manual commit"""
        last_manual = self.get_last_manual_commit()
        if last_manual:
            subprocess.run(["git", "difftool", last_manual, "HEAD"])
```

## Graph Composition with PydanticAI

Building multi-bot workflows by composing simple bots:

```python
from pydantic_graph import Graph, BaseNode, End, GraphRunContext
from dataclasses import dataclass, field

@dataclass
class BotChainState:
    """State passed between bots in a workflow"""
    original_task: str
    context_spec: Optional[ContextSpec]

    # Accumulate outputs from each bot
    all_file_changes: List[FileChange] = field(default_factory=list)
    bot_outputs: List[BotOutput] = field(default_factory=list)

# Transform output of one bot to input of next
class OutputTransformer:
    """Transform bot outputs for next bot in chain"""

    @staticmethod
    def files_to_clipboard(file_changes: List[FileChange]) -> str:
        """Convert file changes to clipboard format for next bot"""
        parts = []
        for fc in file_changes:
            parts.append(f"=== {fc.filepath} ===")
            parts.append(fc.content)
        return "\n".join(parts)

# Node definitions
@dataclass
class WriterNode(BaseNode[BotChainState]):
    """Node that runs test writer bot"""

    async def run(self, ctx: GraphRunContext[BotChainState]) -> "TesterNode":
        bot = TesterBot()
        output = await bot.execute(
            task=ctx.state.original_task,
            context_spec=ctx.state.context_spec
        )

        ctx.state.all_file_changes.extend(output.file_changes)
        ctx.state.bot_outputs.append(output)

        return TesterNode()

@dataclass
class TesterNode(BaseNode[BotChainState]):
    """Node that runs tests"""

    async def run(self, ctx: GraphRunContext[BotChainState]) -> Union["DebuggerNode", End]:
        test_files = [
            fc.filepath for fc in ctx.state.all_file_changes
            if "test" in fc.filepath
        ]

        result = subprocess.run(["pytest"] + test_files, capture_output=True, text=True)

        if result.returncode == 0:
            return End(ctx.state)
        else:
            # Pass test output to debugger via state
            ctx.state.test_output = result.stdout + result.stderr
            return DebuggerNode()

@dataclass
class DebuggerNode(BaseNode[BotChainState]):
    """Node that debugs failures"""

    async def run(self, ctx: GraphRunContext[BotChainState]) -> End:
        bot = DebuggerBot()
        output = await bot.execute(
            task=f"Fix the failing tests",
            clipboard=ctx.state.test_output,
            context_spec=ctx.state.context_spec
        )

        ctx.state.all_file_changes.extend(output.file_changes)
        ctx.state.bot_outputs.append(output)

        return End(ctx.state)

# Create workflow
test_workflow = Graph(
    nodes=[WriterNode, TesterNode, DebuggerNode],
    state_type=BotChainState
)
```

## Summarizer Bot (Example Application)

A practical implementation for documentation needs:

```python
class CodebaseSummary(BaseModel):
    """Structured summary output"""
    overview: str
    sections: List[Dict[str, str]]  # [{title, content, source_files}]
    total_tokens: int

class SummarizerBot(SimpleBot):
    def __init__(self):
        super().__init__(name="summarizer")
        self.agent = Agent(
            model="anthropic:claude-3-opus-20240229",
            output_type=CodebaseSummary,
            system_prompt=self.bot_prompt
        )

    async def summarize_to_tokens(
        self,
        paths: List[str],
        target_tokens: int,
        custom_instructions: Optional[str] = None
    ) -> CodebaseSummary:
        """Summarize codebase to target token size"""
        # Get content via codeclip
        context_spec = ContextSpec(include_paths=paths)
        user_context, bot_context = await self.context_provider.get_contexts(context_spec)

        prompt = f"""
Summarize this codebase to approximately {target_tokens} tokens.

{f"Additional instructions: {custom_instructions}" if custom_instructions else ""}

{user_context}
{bot_context}
        """

        result = await self.agent.run(prompt)
        return result.output
```

## Google Docs Publisher (Standalone Service)

Sits alongside the bot system, not integrated into it:

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

class GoogleDocsPublisher:
    """Publish content to Google Docs"""

    def __init__(self, credentials_path: str):
        self.creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/documents']
        )
        self.service = build('docs', 'v1', credentials=self.creds)

    def create_doc_from_summary(self, title: str, summary: CodebaseSummary) -> str:
        """Create a new Google Doc from a codebase summary"""
        # Create document
        doc = self.service.documents().create(body={'title': title}).execute()
        doc_id = doc['documentId']

        # Format content
        content = f"""# {title}

## Overview
{summary.overview}

"""
        for section in summary.sections:
            content += f"\n## {section['title']}\n{section['content']}\n"

        # Insert content
        requests = [{
            'insertText': {
                'location': {'index': 1},
                'text': content
            }
        }]

        self.service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': requests}
        ).execute()

        return f"https://docs.google.com/document/d/{doc_id}/edit"

# Usage
summarizer = SummarizerBot()
summary = await summarizer.summarize_to_tokens(
    paths=["metta/rl", "metta/agent"],
    target_tokens=5000
)

publisher = GoogleDocsPublisher("credentials.json")
doc_url = publisher.create_doc_from_summary("Metta RL Module Summary", summary)
```

## CLI Interface

Simple command-line interface for bot execution:

```python
import click
import asyncio

@click.group()
def cli():
    """Codebot CLI"""
    pass

@cli.command()
@click.argument('task')
@click.option('--paths', '-p', multiple=True, help='Paths to include in context')
@click.option('--clipboard/--no-clipboard', default=False, help='Include clipboard content')
def test(task, paths, clipboard):
    """Run test writer bot"""
    bot = TesterBot()

    # Get clipboard if requested
    clipboard_content = None
    if clipboard:
        clipboard_content = pyperclip.paste()

    # Run bot
    context_spec = ContextSpec(include_paths=list(paths)) if paths else None
    output = asyncio.run(bot.execute(task, clipboard_content, context_spec))

    # Apply changes
    for change in output.file_changes:
        Path(change.filepath).write_text(change.content)

    click.echo(f"Created/modified {len(output.file_changes)} files")
    click.echo(output.explanation)

@cli.command()
def prompts():
    """List available prompts"""
    manager = PromptManager()
    for name, description in manager.list_prompts().items():
        click.echo(f"{name}: {description}")

@cli.command()
def reset():
    """Reset to last manual commit"""
    git = GitManager()
    git.reset_to_last_manual()

@cli.command()
def diff():
    """Show diff since last manual commit"""
    git = GitManager()
    git.diff_since_last_manual()

if __name__ == '__main__':
    cli()
```
