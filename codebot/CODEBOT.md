# Stage 1: Codebot CLI

Codebot provides AI-powered development assistance through a unified CLI with multiple execution modes.

## Installation

```bash
pip install codebot
export ANTHROPIC_API_KEY=your-key-here
```

## Quick Usage

```bash
# Summarize code for understanding - a useful early command to build
codebot summarize src/api/  # Creates .codebot/summaries/api_summary.md
codebot summarize src/ --max-tokens 5000  # Larger summary

# One-shot commands
codebot test src/main.py
codebot debug-tests metta/rl  # reads test output from clipboard
codebot fix-types src/
codebot review  # reviews current git diff

# Interactive modes
codebot implement -i          # Launch Claude Code
codebot refactor -p          # Pipeline mode (claude -p)
codebot fix -r               # Review changes before applying

# Custom prompts
codebot @"optimize for performance" src/slow.py
```

## Core Architecture

### The Summarizer Pattern: Foundation for Context-Aware AI

One of the most useful early commands to build is the **summarizer** - a command that takes a set of files and produces a token-constrained summary. This pattern is valuable because:

1. **Token Budget Management**: LLMs have context limits; summaries let us fit more information
2. **Hierarchical Understanding**: Summaries can be chained (file → module → project)
3. **Caching Layer**: Summaries can be cached and reused across commands
4. **Human-Readable Context**: Developers can review what context the AI is using

```python
class SummaryCache:
    """Cache summaries to avoid recomputation"""
    
    def __init__(self, cache_dir: Path = Path(".codebot/summaries")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_or_create_summary(self, 
                                   files: List[str], 
                                   max_tokens: int = 2000) -> str:
        """Get cached summary or create new one"""
        
        # Generate cache key from file paths + modification times
        cache_key = self._generate_cache_key(files)
        cache_path = self.cache_dir / f"{cache_key}.md"
        
        if cache_path.exists():
            return cache_path.read_text()
        
        # Create new summary
        summary = await self._create_summary(files, max_tokens)
        cache_path.write_text(summary)
        
        return summary
    
    async def _create_summary(self, files: List[str], max_tokens: int) -> str:
        """Create summary using the summarize command"""
        
        # Gather file contents
        context = ContextManager().gather_context(files)
        
        # Use specialized summary agent
        summarizer = Agent(
            result_type=str,
            system_prompt=f"""Create a summary of the provided code that:
            1. Captures the essential functionality and structure
            2. Preserves important technical details
            3. Stays under {max_tokens} tokens
            4. Is optimized for another LLM to quickly understand the codebase"""
        )
        
        result = await summarizer.run({
            "files": context.files,
            "token_limit": max_tokens
        })
        
        return result.data

# Usage in other commands
class CommandWithSummary(Command):
    """Commands can leverage summaries for better context"""
    
    async def execute(self, context: ExecutionContext) -> CommandOutput:
        # Get summary of related modules
        summary_cache = SummaryCache()
        
        # Summarize test files to understand test patterns
        test_summary = await summary_cache.get_or_create_summary(
            glob.glob("tests/**/*.py"), 
            max_tokens=1000
        )
        
        # Add summary to context
        context.metadata["test_patterns"] = test_summary
        
        # Now execute with enriched context
        return await super().execute(context)
```

#### Building the Summarizer with PydanticAI

Here's how to implement the summarizer using PydanticAI's structured outputs:

```python
from typing import List, Dict, Optional
from pydantic_ai import Agent, ModelRetry
from pydantic import BaseModel, Field, validator

# Define structured models for the summarizer
class CodeComponent(BaseModel):
    """A significant code component identified in the summary"""
    name: str
    type: Literal["class", "function", "module", "interface"]
    description: str
    file_path: str
    dependencies: List[str] = Field(default_factory=list)

class CodePattern(BaseModel):
    """Identified pattern or convention in the codebase"""
    pattern: str
    description: str
    examples: List[str] = Field(default_factory=list)

class SummaryResult(BaseModel):
    """Structured output from code summarization"""
    overview: str = Field(description="High-level overview of the codebase")
    components: List[CodeComponent] = Field(description="Key components identified")
    external_dependencies: List[str] = Field(description="External packages/libraries used")
    patterns: List[CodePattern] = Field(description="Common patterns and conventions")
    entry_points: List[str] = Field(description="Main entry points into the code")
    
    @validator('overview')
    def overview_not_too_long(cls, v):
        # Ensure overview stays concise
        if len(v.split()) > 500:
            raise ValueError("Overview must be under 500 words")
        return v
    
    def to_markdown(self) -> str:
        """Convert to readable markdown format"""
        sections = [
            f"# Code Summary\n\n{self.overview}",
            "\n## Key Components\n" + "\n".join(
                f"- **{c.name}** ({c.type}): {c.description}"
                for c in self.components
            ),
            "\n## Dependencies\n" + "\n".join(f"- {d}" for d in self.external_dependencies),
            "\n## Patterns\n" + "\n".join(
                f"- **{p.pattern}**: {p.description}"
                for p in self.patterns
            )
        ]
        return "\n".join(sections)

# Context model for the agent
class SummaryContext(BaseModel):
    """Context provided to the summarizer agent"""
    files: Dict[str, str] = Field(description="Map of file paths to content")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")
    max_tokens: int = Field(default=2000, description="Target token count for summary")

class SummarizeCommand(Command):
    """Implementation using PydanticAI agents"""
    
    async def execute(self, context: ExecutionContext) -> CommandOutput:
        """Execute summarization using structured PydanticAI agent"""
        
        # Create agent with structured result type
        summarizer = Agent(
            result_type=SummaryResult,
            system_prompt="""Analyze code to create a structured summary optimized for AI consumption.
            Focus on architecture, key components, and patterns that would help another AI understand the codebase quickly."""
        )
        
        # Prepare context using our model
        summary_context = SummaryContext(
            files=context.files,
            focus_areas=self._identify_focus_areas(context)
        )
        
        # Run agent with retry logic
        try:
            result = await summarizer.run(summary_context)
            summary = result.data
            
            # Convert to markdown
            summary_content = summary.to_markdown()
            
            # Create output file
            return CommandOutput(
                file_changes=[FileChange(
                    filepath=".codebot/summaries/latest.md",
                    content=summary_content
                )],
                summary=f"Analyzed {len(context.files)} files, found {len(summary.components)} key components",
                metadata={
                    "component_count": len(summary.components),
                    "dependency_count": len(summary.external_dependencies),
                    "pattern_count": len(summary.patterns)
                }
            )
            
        except ModelRetry as retry:
            # Handle retry with additional context
            self.logger.warning(f"Retrying summary: {retry.message}")
            # Could add more files or adjust parameters here
            raise

# Example of using dependencies for smarter summarization
class HierarchicalSummarizer:
    """Build summaries that reference other summaries"""
    
    def __init__(self):
        self.summary_agent = Agent(
            result_type=SummaryResult,
            system_prompt="Create hierarchical code summaries"
        )
        
        # Agent for merging multiple summaries
        self.merge_agent = Agent(
            result_type=SummaryResult,
            system_prompt="Merge multiple code summaries into a cohesive overview"
        )
    
    async def summarize_module(self, module_path: str, 
                              existing_summaries: Dict[str, SummaryResult]) -> SummaryResult:
        """Summarize a module using existing sub-module summaries"""
        
        context = {
            "module_path": module_path,
            "subsummaries": {
                path: summary.overview 
                for path, summary in existing_summaries.items()
                if path.startswith(module_path)
            }
        }
        
        result = await self.merge_agent.run(context)
        return result.data
```

### File Operations: The Fundamental Unit

All codebot operations produce file changes:

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from pathlib import Path

class ExecutionContext(BaseModel):
    """Context passed to commands for execution"""
    git_diff: str = ""              # Current uncommitted changes
    clipboard: str = ""             # User's clipboard content
    files: Dict[str, str] = {}      # filepath -> content
    working_directory: Path = Field(default_factory=Path.cwd)
    token_count: int = 0
    
    def add_file(self, filepath: str, content: str) -> None:
        """Add file to context"""
        self.files[filepath] = content

class FileChange(BaseModel):
    """Atomic unit of code modification"""
    filepath: str
    content: str
    operation: Literal["write", "delete"] = "write"
    
    def apply(self) -> None:
        """Apply change to filesystem"""
        path = Path(self.filepath)
        if self.operation == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.content)
        elif self.operation == "delete":
            path.unlink(missing_ok=True)
    
    def preview(self) -> str:
        """Generate diff preview"""
        if self.operation == "delete":
            return f"Delete {self.filepath}"
        
        path = Path(self.filepath)
        if path.exists():
            import difflib
            original = path.read_text().splitlines(keepends=True)
            new = self.content.splitlines(keepends=True)
            diff = difflib.unified_diff(original, new, fromfile=self.filepath)
            return "".join(diff)
        else:
            return f"Create {self.filepath}:\n{self.content[:200]}..."

class CommandOutput(BaseModel):
    """Standard output from any command execution"""
    file_changes: List[FileChange]
    summary: str = ""
    metadata: Dict[str, Any] = {}
```

### Context Management

Smart context gathering with token awareness:

```python
import tiktoken
from typing import Tuple, List

class ContextManager:
    """Manages context gathering with token budgeting"""
    
    def __init__(self, token_budget: int = 50000):
        self.token_budget = token_budget
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def gather_context(self, paths: List[str], 
                      include_git: bool = True,
                      include_clipboard: bool = True) -> ExecutionContext:
        """Gather relevant context within token budget"""
        
        context = ExecutionContext()
        tokens_used = 0
        
        # Priority 1: Git diff (if requested)
        if include_git:
            git_diff = self._get_git_diff()
            diff_tokens = len(self.encoder.encode(git_diff))
            if tokens_used + diff_tokens <= self.token_budget:
                context.git_diff = git_diff
                tokens_used += diff_tokens
        
        # Priority 2: Clipboard (for debug commands)
        if include_clipboard:
            clipboard = pyperclip.paste()
            clip_tokens = len(self.encoder.encode(clipboard))
            if tokens_used + clip_tokens <= self.token_budget:
                context.clipboard = clipboard
                tokens_used += clip_tokens
        
        # Priority 3: Target files
        file_scores = self._score_files(paths, context.git_diff)
        for score, filepath, content in sorted(file_scores, reverse=True):
            file_tokens = len(self.encoder.encode(content))
            if tokens_used + file_tokens <= self.token_budget:
                context.add_file(filepath, content)
                tokens_used += file_tokens
        
        context.token_count = tokens_used
        return context
    
    def _score_files(self, patterns: List[str], git_diff: str) -> List[Tuple[float, str, str]]:
        """Score files by relevance
        
        Prioritization order:
        1. Files mentioned in git diff (10x weight)
        2. Test files for test commands (2x weight)
        3. Import relationships (1.5x weight)
        4. Recently modified files (1.2x weight)
        5. Smaller files preferred when equal priority
        """
        scores = []
        diff_files = self._extract_diff_files(git_diff)
        
        for pattern in patterns:
            for filepath in glob.glob(pattern, recursive=True):
                if Path(filepath).is_file():
                    content = Path(filepath).read_text()
                    
                    # Base scoring
                    base_score = 1.0
                    
                    # 1. Files in git diff get highest priority
                    if filepath in diff_files:
                        base_score = 10.0
                    
                    # 2. Boost test files when working on source
                    if "test" in filepath and any("test" not in p for p in patterns):
                        base_score *= 2.0
                    
                    # 3. Check import relationships
                    if self._has_import_relationship(filepath, diff_files):
                        base_score *= 1.5
                    
                    # 4. Recent modification time
                    mtime = Path(filepath).stat().st_mtime
                    if time.time() - mtime < 86400:  # Modified in last day
                        base_score *= 1.2
                    
                    # 5. Prefer smaller files when priorities are similar
                    file_size = len(content)
                    if file_size > 10000:  # Penalize large files
                        base_score *= 0.8
                    
                    scores.append((base_score, filepath, content))
        
        return scores
```

### Unified Agent Execution: Interactive, Pipeline, or Remote

The key is ensuring the same inputs flow to each execution mode:

```python
from typing import TypeVar, Optional, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent
import subprocess
import json

T = TypeVar('T', bound=BaseModel)

async def execute_agent(prompt: str, 
                       context: Dict[str, Any],
                       result_type: type[T],
                       mode: Optional[str] = None) -> T:
    """Execute an agent in the appropriate mode based on environment"""
    
    # Prepare the same input for all modes
    agent_input = {
        "prompt": prompt,
        "context": context,
        "expected_output_type": result_type.__name__
    }
    
    # Determine execution mode
    if mode == "interactive" or (not mode and _claude_code_available()):
        # Hand off to Claude Code
        return await _execute_interactive(agent_input, result_type)
    
    elif mode == "pipeline" or (not mode and os.environ.get('CODEBOT_PIPELINE_ID')):
        # Use pipeline mode
        session_id = os.environ.get('CODEBOT_PIPELINE_ID', 'default')
        return await _execute_pipeline(agent_input, result_type, session_id)
    
    else:
        # Standard remote execution
        return await _execute_remote(agent_input, result_type)

async def _execute_remote(agent_input: Dict[str, Any], result_type: type[T]) -> T:
    """Standard PydanticAI execution"""
    agent = Agent(
        result_type=result_type,
        system_prompt=agent_input["prompt"]
    )
    result = await agent.run(agent_input["context"])
    return result.data

async def _execute_pipeline(agent_input: Dict[str, Any], 
                           result_type: type[T], 
                           session_id: str) -> T:
    """Execute via claude -p (pipeline mode)
    
    Note: Structured output extraction from pipeline mode is an open question.
    This implementation shows one possible approach.
    """
    # Format input for pipeline
    formatted_input = f"""{agent_input["prompt"]}

Context:
{json.dumps(agent_input["context"], indent=2)}
"""
    
    # Run pipeline command
    proc = await asyncio.create_subprocess_exec(
        'claude', '-p', session_id,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )
    
    stdout, _ = await proc.communicate(formatted_input.encode())
    response = stdout.decode()
    
    # TODO: Structured output extraction from pipeline mode
    # For now, attempt basic JSON extraction
    # This is an area for future development
    raise NotImplementedError(
        "Pipeline mode structured output extraction is not yet implemented. "
        "Use remote mode for reliable structured outputs."
    )

async def _execute_interactive(agent_input: Dict[str, Any], result_type: type[T]) -> T:
    """Hand off to Claude Code"""
    import tempfile
    
    # Write context file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(agent_input, f, indent=2)
        context_file = f.name
    
    # Launch Claude Code
    subprocess.run([
        'claude',
        '--task', agent_input["prompt"],
        '--context-file', context_file
    ])
    
    # Interactive mode is for human-in-the-loop
    raise NotImplementedError(
        "Interactive mode launched Claude Code. "
        "For programmatic execution, use mode='remote' or mode='pipeline'"
    )

def _claude_code_available() -> bool:
    """Check if claude CLI is available"""
    try:
        result = subprocess.run(['claude', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Simple command execution
class Command(BaseModel):
    """Executable command"""
    name: str
    prompt_template: str
    default_paths: List[str] = []
    
    async def execute(self, 
                     context: ExecutionContext,
                     mode: Optional[str] = None) -> CommandOutput:
        """Execute command in specified mode"""
        
        # Same inputs regardless of mode
        return await execute_agent(
            prompt=self.prompt_template,
            context=self._format_context(context),
            result_type=CommandOutput,
            mode=mode
        )

    
    def _format_context(self, context: ExecutionContext) -> str:
        """Format context for LLM consumption"""
        parts = []
        
        if context.git_diff:
            parts.append(f"Git changes:\n```diff\n{context.git_diff}\n```")
        
        if context.clipboard:
            parts.append(f"Clipboard (error/context):\n```\n{context.clipboard}\n```")
        
        for filepath, content in context.files.items():
            parts.append(f"File: {filepath}\n```python\n{content}\n```")
        
        return "\n\n".join(parts)
    
    def _validate_output(self, output: CommandOutput) -> bool:
        """Validate command output meets requirements"""
        # Check file changes are valid
        for change in output.file_changes:
            if not change.filepath:
                return False
            if change.operation not in ["write", "delete"]:
                return False
            if change.operation == "write" and not change.content:
                return False
        
        # Check summary exists
        if not output.summary:
            return False
        
        return True
```

### Built-in Commands

A good first command to implement is the summarizer - it's useful for all context-aware operations:

```python
# Command registry with concrete implementations
COMMANDS = {
    "summarize": Command(
        name="summarize",
        prompt_template="""Create a concise summary of the provided files.
        
        Guidelines:
        - Focus on understanding the code's purpose and structure
        - Identify key functions, classes, and their relationships
        - Note important patterns and conventions used
        - Highlight any critical dependencies
        - Keep the summary under the token limit while preserving essential information
        
        The summary should help another LLM quickly understand this codebase.""",
        default_paths=["**/*.py"],
        metadata={"max_output_tokens": 2000}  # Configurable limit
    ),
    
    "test": Command(
        name="test",
        prompt_template="""Write comprehensive tests for the provided code.
        
        Guidelines:
        - Follow the project's test patterns and conventions
        - Include unit tests for all public functions
        - Add edge cases and error scenarios
        - Use descriptive test names
        - Ensure tests are independent and deterministic""",
        default_paths=["**/*.py", "!test_*.py", "!*_test.py"]
    ),
    
    "debug-tests": Command(
        name="debug-tests",
        prompt_template="""Fix the failing tests based on the error output.
        
        The clipboard contains test failure output. Analyze the errors and:
        1. Understand why tests are failing
        2. Fix the implementation (not the tests) unless tests are wrong
        3. Ensure fixes don't break other functionality
        4. Add comments explaining non-obvious fixes""",
        default_paths=["**/*.py"]
    ),
    
    "refactor": Command(
        name="refactor",
        prompt_template="""Refactor code to improve quality without changing behavior.
        
        Focus on:
        - Extracting duplicate code into functions
        - Simplifying complex logic
        - Improving naming and readability
        - Following SOLID principles
        - Reducing cyclomatic complexity""",
        default_paths=["**/*.py"]
    ),
    
    "implement": Command(
        name="implement",
        prompt_template="""Implement code to make the failing tests pass.
        
        Guidelines:
        - Write minimal code to pass tests
        - Follow existing patterns in the codebase
        - Add appropriate error handling
        - Include type hints
        - Document complex logic""",
        default_paths=["**/*.py"]
    )
}
```

## Execution Modes

### Command Mode (Default)

One-shot execution with immediate results:

```python
async def execute_command(cmd_name: str, paths: List[str], **kwargs) -> None:
    """Execute a single command"""
    
    # Get command
    command = COMMANDS.get(cmd_name)
    if not command:
        raise ValueError(f"Unknown command: {cmd_name}")
    
    # Gather context
    context_paths = paths or command.default_paths
    context = ContextManager().gather_context(context_paths)
    
    # Execute
    output = await command.execute(context)
    
    # Handle output based on flags
    if kwargs.get("dry_run"):
        for change in output.file_changes:
            print(change.preview())
    elif kwargs.get("review"):
        output = await review_changes(output)
        apply_changes(output.file_changes)
    else:
        apply_changes(output.file_changes)
    
    print(output.summary)
```

### Loop Mode

Continuous execution toward goals:

```python
class LoopState(BaseModel):
    """Persistent state for continuous loops"""
    name: str
    goal: str
    metrics: Dict[str, float] = {}
    iteration_count: int = 0
    last_run: Optional[datetime] = None
    
class Loop:
    """Continuous execution with goal tracking"""
    
    def __init__(self, command: Command, goal: str):
        self.command = command
        self.state = LoopState(name=command.name, goal=goal)
        
    async def run_iteration(self) -> bool:
        """Run one loop iteration"""
        
        # Measure current state
        current_metrics = await self._measure_metrics()
        self.state.metrics.update(current_metrics)
        
        # Check if goal is met
        if self._goal_achieved():
            return False  # Stop loop
        
        # Execute command
        context = ContextManager().gather_context(self.command.default_paths)
        output = await self.command.execute(context)
        
        # Apply changes
        if output.file_changes:
            apply_changes(output.file_changes)
            self.state.iteration_count += 1
            self.state.last_run = datetime.now()
            
        return True  # Continue loop
```

## Part 2: Interactive Modes

### Overview

Interactive modes provide different levels of human involvement:

- **Claude Code (`-i`)**: Full IDE collaboration
- **Pipeline (`-p`)**: Stateful conversations
- **Review (`-r`)**: Change-by-change approval
- **Guided (`-g`)**: Step-by-step explanation

### Claude Code Mode (`-i`)

Launch Claude Code with intelligently gathered context:

```python
class ClaudeCodeSession:
    """Manages Claude Code sessions with pre-loaded context"""
    
    async def launch(self, command: Command, paths: List[str]) -> None:
        """Launch Claude Code with prepared context"""
        
        # 1. Gather comprehensive context
        context = await self._build_context(paths)
        
        # 2. Generate initial prompt using AI
        prompt_agent = Agent(
            result_type=str,
            system_prompt="""Create a clear, actionable prompt for Claude Code that:
            1. Explains the task clearly
            2. Points out relevant files and context
            3. Suggests a good starting approach
            4. Highlights any constraints or requirements"""
        )
        
        initial_prompt = await prompt_agent.run({
            "command": command.name,
            "base_prompt": command.prompt_template,
            "context_summary": self._summarize_context(context),
            "user_paths": paths
        })
        
        # 3. Launch Claude Code
        subprocess.run([
            "claude",
            "--task", command.name,
            "--context", json.dumps({
                "prompt": initial_prompt.data,
                "files": context.files,
                "git_branch": self._get_current_branch()
            })
        ])
```

### Pipeline Mode (`-p`)

Stateful conversation using claude -p:

```python
class PipelineSession:
    """Manages pipeline conversations with state preservation"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"pipeline_{uuid.uuid4().hex[:8]}"
        self.conversation_history: List[ConversationTurn] = []
    
    async def execute_command(self, 
                            command: Command,
                            context: ExecutionContext) -> CommandOutput:
        """Execute command in pipeline mode with conversation history"""
        
        # Build conversation prompt including history
        conversation_prompt = self._build_conversation_prompt(command, context)
        
        # Execute with claude -p
        proc = await asyncio.create_subprocess_exec(
            'claude', '-p', self.session_id,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate(conversation_prompt.encode())
        response = stdout.decode()
        
        # Parse response with PydanticAI
        parser = Agent(
            result_type=CommandOutput,
            system_prompt="Extract file changes from the response"
        )
        
        output = await parser.run(response)
        
        # Update conversation history
        self.conversation_history.extend([
            ConversationTurn(role="user", content=conversation_prompt),
            ConversationTurn(role="assistant", content=response)
        ])
        
        return output.data
```

### Review Mode (`-r`)

Review mode lets you see and approve/reject each change before it's applied to your files. Think of it like a git diff where you can selectively apply hunks.

#### Example Review Session

```bash
$ codebot fix-types -r src/
Analyzing 5 files for missing type annotations...

[1/12] src/utils.py - Line 23
--- before
+++ after
@@ -23,7 +23,7 @@
-def calculate_score(items, weights):
+def calculate_score(items: list[float], weights: list[float]) -> float:
     return sum(i * w for i, w in zip(items, weights))

AI Explanation: Added type hints to clarify that both parameters are lists of floats
and the function returns a float. This helps catch type errors at development time.

Options: [a]pprove, [r]eject, [m]odify, [s]kip remaining, [A]pprove all similar
Your choice: a
✓ Approved

[2/12] src/utils.py - Line 45  
--- before
+++ after
@@ -45,3 +45,3 @@
-def process_data(data):
+def process_data(data: Any) -> None:
     # Complex processing...

AI Explanation: Added 'Any' type since the data structure is unclear from context.
Consider using a more specific type like Dict or a custom dataclass.

Options: [a]pprove, [r]eject, [m]odify, [s]kip remaining, [A]pprove all similar  
Your choice: m
Describe the modification needed: > Use Dict[str, Any] instead of Any
✓ Modified

[Summary]
Reviewed: 12 changes
Approved: 8  
Rejected: 2
Modified: 2

Applying approved changes...
✓ Done
```

#### Implementation

```python
class ReviewSession:
    """Manages interactive review of AI-proposed changes"""
    
    async def review_changes(self, 
                           command: Command,
                           output: CommandOutput) -> CommandOutput:
        """Interactively review each proposed change"""
        
        reviewed_changes = []
        
        for i, change in enumerate(output.file_changes, 1):
            # Show diff with syntax highlighting
            diff = change.preview()
            self._display_diff_with_syntax(diff, change.filepath)
            
            # Get AI explanation
            if self.use_ai_explanations:
                explanation = await self._explain_change(change, command)
                print(f"\nAI Explanation: {explanation}")
            
            # Get human decision
            decision = await self._get_decision(change)
            
            if decision.decision == "approve":
                reviewed_changes.append(change)
            elif decision.decision == "modify":
                modified = await self._apply_modification(change, decision.modification)
                reviewed_changes.append(modified)
        
        return CommandOutput(
            file_changes=reviewed_changes,
            summary=f"Applied {len(reviewed_changes)} of {len(output.file_changes)} changes"
        )
```

### Guided Mode (`-g`)

AI guides through multi-step operations:

```python
class GuidedSession:
    """AI guides human through complex workflows step-by-step"""
    
    async def execute_workflow(self, 
                             workflow_name: str,
                             initial_context: Dict[str, Any]) -> List[CommandOutput]:
        """Execute a workflow with step-by-step guidance"""
        
        workflow_agent = Agent(
            result_type=WorkflowStep,
            system_prompt="Guide the user through development workflows"
        )
        
        results = []
        context = initial_context
        
        while True:
            # Get next step from AI
            step = await workflow_agent.run({
                "workflow": workflow_name,
                "context": context,
                "completed": results
            })
            
            if step.data.is_complete:
                break
            
            # Present step to user
            print(f"\nStep {step.data.number}: {step.data.title}")
            print(f"Rationale: {step.data.rationale}")
            
            if not self._get_approval():
                continue
            
            # Execute step
            output = await self._execute_step(step.data)
            results.append(output)
            context["last_output"] = output.summary
        
        return results
```

## CLI Implementation

```python
import click
import asyncio

@click.command()
@click.argument('action')
@click.argument('paths', nargs=-1)
@click.option('-r', '--review', is_flag=True, help='Review changes')
@click.option('-l', '--loop', is_flag=True, help='Run as loop')
@click.option('--goal', help='Goal for loop mode')
@click.option('--dry-run', is_flag=True, help='Preview only')
@click.option('-i', '--interactive', is_flag=True, help='Claude Code')
@click.option('-p', '--pipeline', is_flag=True, help='Pipeline mode')
def codebot(action: str, paths: tuple, **kwargs):
    """AI-powered development assistant"""
    
    async def run():
        if action.startswith("@"):
            # Custom prompt
            command = Command(
                name="custom",
                prompt_template=action[1:],
                default_paths=list(paths) or ["**/*.py"]
            )
        else:
            command = COMMANDS.get(action)
            if not command:
                click.echo(f"Unknown command: {action}")
                return
        
        if kwargs['loop']:
            # Loop mode
            if not kwargs.get('goal'):
                click.echo("--goal required for loop mode")
                return
            
            loop = Loop(command, kwargs['goal'])
            while await loop.run_iteration():
                await asyncio.sleep(60)  # Wait between iterations
        
        elif kwargs['interactive'] or kwargs['pipeline']:
            # Delegate to interactive modes
            from .interactive import launch_interactive
            await launch_interactive(command, paths, **kwargs)
        
        else:
            # Standard command execution
            await execute_command(action, list(paths), **kwargs)
    
    asyncio.run(run())

if __name__ == '__main__':
    codebot()
```

## Working with PydanticAI Models

The key to building effective codebot commands is defining good Pydantic models:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal
from pydantic_ai import Agent

# Model for test generation
class TestCase(BaseModel):
    """A single test case to generate"""
    test_name: str = Field(description="Descriptive test function name")
    test_type: Literal["unit", "integration", "edge_case"]
    imports_needed: List[str] = Field(default_factory=list)
    test_code: str = Field(description="Complete test function code")
    
    @validator('test_name')
    def valid_python_name(cls, v):
        if not v.startswith('test_'):
            v = f'test_{v}'
        return v.replace(' ', '_').lower()

class TestGenerationResult(BaseModel):
    """Result of test generation"""
    test_cases: List[TestCase]
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    additional_fixtures: List[str] = Field(default_factory=list)
    
    def to_test_file(self) -> str:
        """Convert to a complete test file"""
        imports = set()
        for test in self.test_cases:
            imports.update(test.imports_needed)
        
        sections = [
            "import pytest",
            *sorted(imports),
            "",
            self.setup_code or "",
            *[test.test_code for test in self.test_cases],
            self.teardown_code or ""
        ]
        
        return "\n\n".join(filter(bool, sections))

# Using models with agents
class TestCommand(Command):
    async def execute(self, context: ExecutionContext) -> CommandOutput:
        agent = Agent(
            result_type=TestGenerationResult,
            system_prompt="Generate comprehensive tests following project conventions"
        )
        
        result = await agent.run({
            "code_to_test": context.files,
            "existing_tests": self._find_existing_tests(context)
        })
        
        test_file_content = result.data.to_test_file()
        
        return CommandOutput(
            file_changes=[FileChange(
                filepath="tests/test_generated.py",
                content=test_file_content
            )],
            summary=f"Generated {len(result.data.test_cases)} test cases"
        )
```

## Extending Codebot

### Adding Commands

```python
# ~/.codebot/commands.py

from codebot import Command, register_command

@register_command
class SecurityAudit(Command):
    name = "security-audit"
    prompt_template = """Analyze code for security vulnerabilities.
    
    Check for:
    - SQL injection risks
    - XSS vulnerabilities  
    - Insecure cryptography
    - Hardcoded secrets
    - Authentication flaws"""
    
    default_paths = ["**/*.py", "**/*.js"]
    require_confirmation = True
```

### Custom Context Providers

```python
class TestResultsProvider(ContextProvider):
    """Include recent test results in context"""
    
    def gather(self, paths: List[str]) -> Dict[str, Any]:
        results = {}
        
        # Find pytest cache
        cache_dir = Path(".pytest_cache/v/cache")
        if cache_dir.exists():
            results["last_failed"] = (cache_dir / "lastfailed").read_text()
        
        # Find coverage report
        if Path("htmlcov/index.html").exists():
            results["coverage_summary"] = self._parse_coverage()
        
        return results
```

## Git Integration & Version Control

### Automatic Commit Management

```python
class GitManager:
    """Manages git operations and history tracking"""
    
    def __init__(self, auto_commit: bool = True):
        self.auto_commit = auto_commit
        self.commit_prefix = "[codebot]"
    
    async def apply_changes_with_commit(self, 
                                      changes: List[FileChange],
                                      command: str,
                                      summary: str) -> str:
        """Apply changes and create atomic commit"""
        
        # Create branch for changes
        branch_name = f"codebot/{command}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self._create_branch(branch_name)
        
        try:
            # Apply all changes
            for change in changes:
                change.apply()
            
            # Stage changes
            await self._git_add([c.filepath for c in changes])
            
            # Commit with structured message
            commit_msg = f"{self.commit_prefix} {command}: {summary}\n\n"
            commit_msg += "Changes:\n"
            for change in changes:
                commit_msg += f"- {change.operation} {change.filepath}\n"
            
            commit_hash = await self._git_commit(commit_msg)
            
            # Store metadata for rollback
            await self._store_commit_metadata(commit_hash, {
                "command": command,
                "changes": [c.model_dump() for c in changes],
                "timestamp": datetime.now().isoformat()
            })
            
            return commit_hash
            
        except Exception as e:
            # Rollback on failure
            await self._git_reset_hard("HEAD")
            raise
    
    async def rollback_command(self, 
                             command_id: Optional[str] = None,
                             commit_hash: Optional[str] = None) -> bool:
        """Rollback changes from a specific command execution"""
        
        if command_id:
            # Find commit by command ID
            commit_hash = await self._find_commit_by_command(command_id)
        
        if not commit_hash:
            return False
        
        # Create rollback commit
        await self._git_revert(commit_hash)
        return True
```

### Command History Tracking

```python
class CommandHistory(BaseModel):
    """Track all command executions with git integration"""
    command_id: str
    command_name: str
    timestamp: datetime
    git_commit: Optional[str]
    changes_count: int
    rollback_available: bool = True
    
    @classmethod
    def from_execution(cls, cmd: str, output: CommandOutput, commit: str) -> "CommandHistory":
        return cls(
            command_id=f"{cmd}_{uuid.uuid4().hex[:8]}",
            command_name=cmd,
            timestamp=datetime.now(),
            git_commit=commit,
            changes_count=len(output.file_changes)
        )

# CLI integration
@click.command()
@click.option('--history', is_flag=True, help='Show command history')
@click.option('--rollback', help='Rollback command by ID')
def codebot_history(history: bool, rollback: Optional[str]):
    """Manage command history and rollbacks"""
    
    if history:
        # Show recent commands with git commits
        for entry in get_command_history(limit=10):
            print(f"{entry.timestamp}: {entry.command_name} ({entry.command_id})")
            print(f"  Commit: {entry.git_commit[:8]} - {entry.changes_count} files")
    
    elif rollback:
        # Rollback specific command
        if rollback_command(rollback):
            print(f"Successfully rolled back command: {rollback}")
        else:
            print(f"Failed to rollback: {rollback}")
```

## Best Practices

1. **Start with dry-run**: Always preview changes first
2. **Use specific paths**: Target relevant files to stay within token budget
3. **Copy errors to clipboard**: For `debug` commands
4. **Review critical changes**: Use `-r` for important modifications
5. **Set clear goals**: For loop mode, use measurable objectives
6. **Enable git tracking**: Use `--git` flag for automatic commit management
7. **Review history regularly**: Use `codebot --history` to track changes

## See Also

- [WORKFLOWS.md](WORKFLOWS.md) - Stage 2: Multi-step task composition
- [MANYBOT.md](MANYBOT.md) - Stage 3: Autonomous agents with goals