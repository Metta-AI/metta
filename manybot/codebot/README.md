# Codebot

```bash
# Fix failing tests from clipboard
codebot debug-tests

# Interactive development
codebot refactor -i src/api.py

# Pipeline mode with structured output
codebot implement -p src/feature.py
```

## Foundation for AI-Powered Development

Codebot provides AI-powered development assistance through a unified CLI with multiple execution modes, built on PydanticAI for structured agent operations.

### Core Architecture

- **PydanticAI Agents**: Structured AI operations with type-safe inputs/outputs
- **File Operations**: Atomic FileChange as the fundamental unit
- **Context Management**: Token-aware file gathering with smart prioritization  
- **Execution Modes**: Unified interface for oneshot, claudesdk, and interactive execution

### The Summarizer Pattern: Foundation for Context-Aware AI

One of the most useful early commands to build is the **summarizer** - a command that takes a set of files and produces a token-constrained summary. This pattern is valuable because:

1. **Token Budget Management**: LLMs have context limits; summaries let us fit more information
2. **Hierarchical Understanding**: Summaries can be chained (file → module → project)
3. **Caching Layer**: Summaries can be cached and reused across commands
4. **Human-Readable Context**: Developers can review what context the AI is using

## Building the Summarizer with PydanticAI

Here's how to implement the summarizer using PydanticAI's structured outputs:

```python
from typing import List, Dict, Optional, Literal
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

class SummarizeCommand(Command):
    """Implementation using PydanticAI agents"""
    
    async def execute(self, context: ExecutionContext, token_limit: int = 2000) -> CommandOutput:
        """Execute summarization using structured PydanticAI agent"""
        
        # Create agent with structured result type
        summarizer = Agent(
            result_type=SummaryResult,
            system_prompt=f"""Analyze code to create a structured summary optimized for AI consumption.
            Focus on architecture, key components, and patterns that would help another AI understand the codebase quickly.
            Keep the total summary under {token_limit} tokens."""
        )
        
        # Prepare context
        summary_context = {
            "files": context.files,
            "focus_areas": self._identify_focus_areas(context)
        }
        
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
            raise
```

## Summary Cache Pattern

```python
class SummaryCache:
    """Cache summaries to avoid recomputation"""
    
    def __init__(self, cache_dir: Path = Path(".codebot/summaries")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_or_create_summary(self, 
                                   files: List[str], 
                                   token_limit: int = 2000) -> str:
        """Get cached summary or create new one"""
        
        # Generate cache key from file paths + modification times + token limit
        cache_key = self._generate_cache_key(files, token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"
        
        if cache_path.exists():
            return cache_path.read_text()
        
        # Create new summary using PydanticAI
        summarizer = Agent(
            result_type=SummaryResult,
            system_prompt=f"""Create a summary of the provided code that:
            1. Captures the essential functionality and structure
            2. Preserves important technical details
            3. Stays under {token_limit} tokens
            4. Is optimized for another LLM to quickly understand the codebase"""
        )
        
        result = await summarizer.run({
            "files": self._gather_file_contents(files),
            "token_limit": token_limit
        })
        
        summary = result.data.to_markdown()
        cache_path.write_text(summary)
        
        return summary
```

## Core Data Models

```python
class FileChange(BaseModel):
    """Atomic unit of code modification"""
    filepath: str
    content: str
    operation: Literal["write", "delete"] = "write"
    
    def apply(self) -> None:
        """Apply change to filesystem"""
    
    def preview(self) -> str:
        """Generate diff preview"""

class ExecutionContext(BaseModel):
    """Context passed to commands"""
    git_diff: str = ""
    clipboard: str = ""
    files: Dict[str, str] = {}
    working_directory: Path
    token_count: int = 0

class CommandOutput(BaseModel):
    """Standard output from any command"""
    file_changes: List[FileChange]
    summary: str = ""
    metadata: Dict[str, Any] = {}
```

## Command API with PydanticAI

```python
from pydantic_ai import Agent
from typing import TypeVar, Optional

T = TypeVar('T', bound=BaseModel)

class Command(BaseModel):
    """Base class for AI-powered commands using PydanticAI agents"""
    name: str
    prompt_template: str
    default_paths: List[str] = []
    result_type: type[BaseModel] = CommandOutput
    
    def build_agent(self) -> Agent[T]:
        """Build PydanticAI agent for this command"""
        return Agent(
            result_type=self.result_type,
            system_prompt=self.prompt_template
        )
    
    async def execute(self, 
                     context: ExecutionContext,
                     mode: Optional[str] = None) -> CommandOutput:
        """Execute command using appropriate mode"""
        
        # Build agent
        agent = self.build_agent()
        
        # Prepare context
        agent_context = self._format_context(context)
        
        # Execute based on mode
        if mode == "interactive":
            return await self._execute_interactive(agent_context)
        elif mode == "claudesdk":
            return await self._execute_claudesdk(agent, agent_context)
        else:
            # Default oneshot execution with PydanticAI
            result = await agent.run(agent_context)
            return self._process_result(result.data)
    
    def _format_context(self, context: ExecutionContext) -> Dict[str, Any]:
        """Format context for agent consumption"""
        formatted = {
            "files": context.files,
            "working_directory": str(context.working_directory)
        }
        
        if context.git_diff:
            formatted["git_diff"] = context.git_diff
        
        if context.clipboard:
            formatted["error_output"] = context.clipboard
            
        return formatted

# Example: Test generation command with custom result type
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

class TestCommand(Command):
    """Generate comprehensive tests using PydanticAI"""
    name = "test"
    prompt_template = """Generate comprehensive tests following project conventions.
    Include edge cases, error scenarios, and both positive and negative test cases."""
    result_type = TestGenerationResult
    
    async def _process_result(self, result: TestGenerationResult) -> CommandOutput:
        """Convert test generation result to file changes"""
        test_file_content = result.to_test_file()
        
        # Determine output path
        test_path = self._determine_test_path(self.context)
        
        return CommandOutput(
            file_changes=[FileChange(
                filepath=test_path,
                content=test_file_content
            )],
            summary=f"Generated {len(result.test_cases)} test cases",
            metadata={
                "test_types": Counter(tc.test_type for tc in result.test_cases)
            }
        )
```

## Context Management

```python
class ContextManager:
    """Smart context gathering using codeclip"""
    
    def gather_context(self, paths: List[str]) -> ExecutionContext:
        """Gather files using codeclip with intelligent prioritization:
        
        1. Git diff files (10x weight)
        2. Test files for source (2x weight)  
        3. Import relationships (1.5x weight)
        4. Recently modified (1.2x weight)
        5. Smaller files when equal priority
        """
        # Leverages existing codeclip implementation
        from codeclip import get_context
        content, token_info = get_context(paths=paths)
        
        return ExecutionContext(
            files=self._parse_files_from_content(content),
            token_count=token_info["total_tokens"],
            git_diff=self._get_git_diff(),
            clipboard=self._get_clipboard()
        )
```

## Execution Modes

```python
async def execute_agent(prompt: str, 
                       context: Dict[str, Any],
                       result_type: type[T],
                       mode: Optional[str] = None) -> T:
    """Unified execution across modes
    
    - oneshot: PydanticAI structured output (default)
    - claudesdk: Claude SDK mode via claude -p
    - interactive: Launch Claude Code
    """
    
    # Same inputs flow to all modes
    agent_input = {
        "prompt": prompt,
        "context": context,
        "expected_output_type": result_type.__name__
    }
    
    if mode == "oneshot":
        return await _execute_oneshot(agent_input, result_type)
    elif mode == "claudesdk":
        return await _execute_claudesdk(agent_input, result_type)
    elif mode == "interactive":
        return await _execute_interactive(agent_input, result_type)
```

## PR Submission

All code changes can be submitted as pull requests for human review:

```python
class PRSubmission(BaseModel):
    """Submit changes as a pull request"""
    title: str
    description: str
    branch_name: str
    file_changes: List[FileChange]
    base_branch: str = "main"
    draft: bool = False
    
class SubmitPRCommand(Command):
    """Submit file changes as a PR for review"""
    name = "submit-pr"
    
    async def execute(self, 
                     context: ExecutionContext,
                     title: str,
                     description: str = "") -> CommandOutput:
        """Create PR with accumulated changes"""
        
        # 1. Create branch
        branch_name = self._generate_branch_name(title)
        await self._create_branch(branch_name)
        
        # 2. Apply changes to branch
        for change in context.pending_changes:
            change.apply()
        
        # 3. Commit changes
        commit_message = f"{title}\n\n{description}"
        await self._commit_changes(commit_message)
        
        # 4. Push branch
        await self._push_branch(branch_name)
        
        # 5. Create PR
        pr_url = await self._create_pull_request(
            title=title,
            body=self._format_pr_body(description, context),
            branch=branch_name
        )
        
        return CommandOutput(
            file_changes=[],  # Changes are now in PR
            summary=f"Created PR: {pr_url}",
            metadata={"pr_url": pr_url, "branch": branch_name}
        )

# Usage
codebot test src/api.py
codebot refactor src/api.py
codebot submit-pr --title "Add tests and refactor API" --description "Improves test coverage and code quality"
```

## Built-in Commands Using PydanticAI

### Command Registry

```python
# Command registry with concrete PydanticAI implementations
COMMANDS = {
    "summarize": SummarizeCommand(
        name="summarize",
        prompt_template="""Create a concise summary of the provided files.
        Focus on understanding the code's purpose and structure.
        Identify key functions, classes, and their relationships.""",
        default_paths=["**/*.py"],
        result_type=SummaryResult,
        metadata={"default_token_limit": 2000}
    ),
    
    "test": TestCommand(
        name="test",
        prompt_template="""Write comprehensive tests for the provided code.
        Follow the project's test patterns and conventions.
        Include unit tests for all public functions.
        Add edge cases and error scenarios.""",
        default_paths=["**/*.py", "!test_*.py", "!*_test.py"],
        result_type=TestGenerationResult
    ),
    
    "debug-tests": DebugTestsCommand(
        name="debug-tests",
        prompt_template="""Fix the failing tests based on the error output.
        The clipboard contains test failure output. Analyze the errors and:
        1. Understand why tests are failing
        2. Fix the implementation (not the tests) unless tests are wrong
        3. Ensure fixes don't break other functionality""",
        default_paths=["**/*.py"],
        result_type=DebugResult
    ),
    
    "refactor": RefactorCommand(
        name="refactor",
        prompt_template="""Refactor code to improve quality without changing behavior.
        Focus on: extracting duplicate code, simplifying logic, improving naming.""",
        default_paths=["**/*.py"],
        result_type=RefactoringResult
    )
}
```

### Example: Debug Tests Command

```python
class TestFailure(BaseModel):
    """Information about a failing test"""
    test_name: str
    error_type: str
    error_message: str
    traceback: str
    suspected_cause: str

class DebugResult(BaseModel):
    """Result of debugging test failures"""
    failures_analyzed: List[TestFailure]
    fixes_proposed: List[CodeFix]
    root_cause_analysis: str
    
class DebugTestsCommand(Command):
    """Debug failing tests using error output from clipboard"""
    
    async def execute(self, context: ExecutionContext) -> CommandOutput:
        """Execute with special handling for clipboard errors"""
        
        # Create specialized agent for debugging
        debug_agent = Agent(
            result_type=DebugResult,
            system_prompt="""You are an expert at debugging test failures.
            Analyze the error output and determine the root cause.
            Propose fixes to the implementation, not the tests."""
        )
        
        # Include error context from clipboard
        debug_context = {
            "test_output": context.clipboard,
            "source_files": context.files,
            "recent_changes": context.git_diff
        }
        
        result = await debug_agent.run(debug_context)
        debug_data = result.data
        
        # Convert fixes to file changes
        file_changes = []
        for fix in debug_data.fixes_proposed:
            file_changes.append(FileChange(
                filepath=fix.file_path,
                content=fix.new_content,
                metadata={"fix_reason": fix.reason}
            ))
        
        return CommandOutput(
            file_changes=file_changes,
            summary=f"Fixed {len(debug_data.failures_analyzed)} test failures",
            metadata={
                "root_cause": debug_data.root_cause_analysis,
                "failures_fixed": [f.test_name for f in debug_data.failures_analyzed]
            }
        )
```

## CLI Usage

```bash
# Direct execution
codebot test src/main.py

# Work on changes and submit PR when done
codebot test src/api.py
codebot refactor src/api.py
codebot submit-pr --title "Improve API tests and structure"

# Interactive modes
codebot refactor -i     # Claude Code
codebot fix --claudesdk  # Claude SDK mode (claude -p)
codebot lint -r         # Review changes

# Custom prompts
codebot @"optimize for performance" src/slow.py

# Dry run and preview
codebot refactor --dry-run src/
```

### Execution Mode Details

```bash
# Interactive mode - launches Claude Code
codebot refactor -i src/api.py
codebot implement -i --context src/

# Claude SDK mode - autonomous execution via claude -p
codebot fix --claudesdk src/
codebot debug-tests --claudesdk

# Review mode - approve each change
codebot refactor -r src/
codebot fix-types -r --verbose

# Dry run - preview without applying
codebot refactor --dry-run src/
```

## Implementation Examples

### Building Custom Commands

```python
@register_command
class PerformanceOptimizer(Command):
    name = "optimize-perf"
    prompt_template = """Analyze code for performance bottlenecks and suggest optimizations.
    Focus on: algorithmic improvements, caching opportunities, query optimization."""
    result_type = OptimizationResult
    
    async def post_process(self, result: OptimizationResult) -> CommandOutput:
        """Custom post-processing"""
        # Generate report
        report_path = ".codebot/reports/performance.md"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(result.to_report())
        
        # Apply safe optimizations automatically
        file_changes = []
        for opt in result.optimizations:
            if opt.risk_level == "low" and opt.auto_applicable:
                file_changes.extend(opt.generate_changes())
        
        return CommandOutput(
            file_changes=file_changes,
            summary=f"Found {len(result.optimizations)} optimizations",
            metadata={"report": report_path}
        )
```

### Review Mode Implementation

```python
class InteractiveReviewer:
    """Interactive change review with enhanced UI"""
    
    async def review_change(self, change: FileChange) -> ReviewDecision:
        """Review single change with context"""
        
        # Show enhanced diff
        self._display_enhanced_diff(change)
        
        # Show AI explanation if available
        if change.metadata.get("ai_explanation"):
            print(f"\nAI: {change.metadata['ai_explanation']}")
        
        # Get user input
        while True:
            choice = input("\n[a]pprove, [r]eject, [m]odify, [A]pprove pattern: ")
            
            if choice == "a":
                return ReviewDecision(action="approve")
            elif choice == "r":
                reason = input("Reason for rejection: ")
                return ReviewDecision(action="reject", reason=reason)
            elif choice == "m":
                modification = input("Describe modification: ")
                return ReviewDecision(action="modify", modification=modification)
            elif choice == "A":
                pattern = self._create_pattern(change)
                self.approved_patterns.append(pattern)
                return ReviewDecision(action="approve", pattern=pattern)
```

### Claude SDK Execution

```python
class ClaudeSDKExecutor:
    """Execute commands using claude -p"""
    
    def execute_with_claudesdk(self, 
                              command: str,
                              context: ExecutionContext) -> CommandOutput:
        """Execute using Claude SDK (claude -p)"""
        
        # Build command with context
        full_prompt = self._build_prompt(command, context)
        cmd = ["claude", "-p", full_prompt]
        
        # Add output format for structured parsing
        cmd.extend(["--output-format", "json"])
        
        # Execute claude -p (runs to completion autonomously)
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = json.loads(result.stdout)
        
        # Log the explanation for debugging
        logger.info(f"Claude SDK response: {output.get('result', '')[:200]}...")
        
        # Extract the actual file changes that were made
        # The printed explanation is just for logging, not control flow
        return self._extract_file_changes_from_execution(output)
```

## Advanced Patterns

### Chaining Commands

```python
async def chain_commands(*commands: List[tuple[str, dict]]) -> List[CommandOutput]:
    """Execute commands in sequence, passing context forward"""
    outputs = []
    accumulated_context = {}
    
    for cmd_name, cmd_args in commands:
        # Get command
        command = get_command(cmd_name)
        
        # Merge contexts
        context = {**accumulated_context, **cmd_args.get("context", {})}
        
        # Execute
        output = await command.execute(ExecutionContext(**context))
        outputs.append(output)
        
        # Accumulate context
        accumulated_context["previous_output"] = output.summary
        accumulated_context["files_changed"] = [
            fc.filepath for fc in output.file_changes
        ]
    
    return outputs

# Usage
results = await chain_commands(
    ("analyze", {"paths": ["src/"]}),
    ("optimize-perf", {"threshold": 0.1}),
    ("test", {"only_changed": True})
)
```

## Integration Points

Codebot serves as the foundation for:

- **Goalbot**: Uses codebot commands in goal-oriented loops
- **Remotebot**: Deploys codebot operations across environments
- **Manybot**: Coordinates multiple agents using codebot primitives

## Workflow Composition & Pipelining

### Workflow Models

```python
class WorkflowStep(BaseModel):
    """Single step in a workflow"""
    id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    name: str
    command: Command
    
    # Conditional execution
    condition: Optional[str] = None  # Python expression
    on_success: Optional[str] = None  # Next step ID
    on_failure: Optional[str] = None  # Next step ID
    
    # Parallel execution
    parallel_with: List[str] = []  # Step IDs to run in parallel
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on condition"""
        if not self.condition:
            return True
        return eval(self.condition, {"__builtins__": {}}, context)

class Workflow(BaseModel):
    """Composable workflow definition"""
    name: str
    description: str
    steps: Dict[str, WorkflowStep]  # id -> step
    entry_point: str  # First step ID
    
    # Execution settings
    max_parallel: int = 5
    stop_on_failure: bool = True
```

### Built-in Workflows

```python
# Test-Driven Development workflow
TDD_WORKFLOW = Workflow(
    name="tdd",
    description="Test-driven development workflow",
    steps={
        "write_test": WorkflowStep(
            name="Write failing test",
            command=COMMANDS["test"],
            on_success="run_test"
        ),
        "run_test": WorkflowStep(
            name="Verify test fails",
            command=RunTestsCommand(),
            on_success="implement",
            on_failure="fix_test"
        ),
        "implement": WorkflowStep(
            name="Implement feature",
            command=COMMANDS["implement"],
            on_success="verify_pass"
        ),
        "verify_pass": WorkflowStep(
            name="Verify tests pass",
            command=RunTestsCommand(),
            on_success="refactor",
            on_failure="debug"
        ),
        "refactor": WorkflowStep(
            name="Refactor implementation",
            command=COMMANDS["refactor"]
        )
    },
    entry_point="write_test"
)
```

### Workflow Executor

```python
class WorkflowExecutor:
    """Execute workflows with parallel and conditional support"""
    
    async def execute(self, 
                     workflow: Workflow,
                     initial_context: ExecutionContext) -> WorkflowResult:
        """Execute workflow steps with intelligent orchestration"""
        
        pending_steps = {workflow.entry_point}
        completed_steps = set()
        step_outputs = {}
        
        while pending_steps:
            # Execute ready steps in parallel
            ready = self._get_ready_steps(pending_steps, completed_steps)
            
            # Execute up to max_parallel steps
            tasks = []
            for step_id in list(ready)[:workflow.max_parallel]:
                step = workflow.steps[step_id]
                task = asyncio.create_task(
                    self._execute_step(step, initial_context)
                )
                tasks.append((step_id, task))
                pending_steps.remove(step_id)
            
            # Wait for completion
            for step_id, task in tasks:
                output = await task
                step_outputs[step_id] = output
                completed_steps.add(step_id)
                
                # Add next steps based on result
                step = workflow.steps[step_id]
                if output.success and step.on_success:
                    pending_steps.add(step.on_success)
                elif not output.success and step.on_failure:
                    pending_steps.add(step.on_failure)
        
        return WorkflowResult(
            workflow=workflow.name,
            outputs=step_outputs,
            success=all(o.success for o in step_outputs.values())
        )
```

### Pipeline Composition

```python
class PipelineComposer:
    """Compose multiple workflows into pipelines"""
    
    def sequence(self, *workflows: Workflow) -> Workflow:
        """Chain workflows sequentially"""
        combined = Workflow(
            name=f"sequence_{'_'.join(w.name for w in workflows)}",
            description="Sequential workflow composition",
            steps={},
            entry_point=""
        )
        
        last_exit = None
        for workflow in workflows:
            # Prefix step IDs to avoid conflicts
            prefix = f"{workflow.name}_"
            
            # Add all steps with prefixed IDs
            for step_id, step in workflow.steps.items():
                new_id = f"{prefix}{step_id}"
                new_step = step.model_copy()
                
                # Update internal references
                if step.on_success:
                    new_step.on_success = f"{prefix}{step.on_success}"
                if step.on_failure:
                    new_step.on_failure = f"{prefix}{step.on_failure}"
                
                combined.steps[new_id] = new_step
            
            # Connect to previous workflow
            if last_exit:
                combined.steps[last_exit].on_success = f"{prefix}{workflow.entry_point}"
            else:
                combined.entry_point = f"{prefix}{workflow.entry_point}"
            
            # Find exit point
            last_exit = self._find_exit_point(workflow, prefix)
        
        return combined
    
    def parallel(self, *workflows: Workflow) -> Workflow:
        """Run workflows in parallel"""
        # Create a workflow with all entry points in parallel_with
        entry_steps = []
        all_steps = {}
        
        for workflow in workflows:
            prefix = f"{workflow.name}_"
            entry_steps.append(f"{prefix}{workflow.entry_point}")
            
            # Add all steps
            for step_id, step in workflow.steps.items():
                all_steps[f"{prefix}{step_id}"] = step.model_copy()
        
        # Create entry point that triggers all workflows
        entry = WorkflowStep(
            name="Parallel execution",
            command=NoOpCommand(),
            parallel_with=entry_steps[1:]
        )
        
        all_steps["parallel_entry"] = entry
        all_steps.update({s: workflow.steps[s] for s in entry_steps})
        
        return Workflow(
            name=f"parallel_{'_'.join(w.name for w in workflows)}",
            steps=all_steps,
            entry_point="parallel_entry"
        )
```

## Architecture Principles

1. **File Changes as Atoms**: All operations produce FileChange objects
2. **Context is King**: Smart context gathering enables better AI results
3. **Structured I/O**: PydanticAI models ensure reliability
4. **Mode Agnostic**: Same inputs work across all execution modes
5. **Composable Workflows**: Build complex operations from simple steps
6. **Extensible**: Easy to add new commands and providers

## Implementation Roadmap

### Phase 1: Core Foundation
- **Data Models**: `FileChange`, `ExecutionContext`, `CommandOutput` base models
- **Context Management**: Integration with existing codeclip for file gathering
- **File Operations**: `FileChange.apply()`, `FileChange.preview()` methods
- **Summarizer Command**: First PydanticAI agent with `SummaryResult` model and caching

### Phase 2: Command Framework
- **Base Classes**: `Command` abstract class with PydanticAI agent builder
- **Command Registry**: Dictionary-based command registration system
- **Core Commands**: `test`, `debug-tests`, `refactor` with custom result types
- **PR Submission**: `SubmitPRCommand` with git operations and PR creation
- **CLI Router**: Click-based CLI with command discovery and argument parsing

### Phase 3: Execution Modes
- **Oneshot Mode**: Default PydanticAI `Agent.run()` with structured output
- **Claude SDK Mode**: Integration via `claude -p` with JSON output parsing
- **Interactive Mode**: Claude Code launcher with context preparation
- **Review Mode**: `InteractiveReviewer` class with change approval workflow

### Phase 4: Workflow Composition
- **Workflow Models**: `WorkflowStep`, `Workflow` with conditional execution
- **Workflow Executor**: Async execution engine with parallel step support
- **Pipeline Composer**: `sequence()` and `parallel()` workflow combinators
- **Built-in Workflows**: `TDD_WORKFLOW`, feature development workflow definitions

### Phase 5: Advanced Features
- **Error Handling**: `ModelRetry` integration, graceful failure modes
- **Command Chaining**: `chain_commands()` with context accumulation
- **Token Optimization**: Summary caching, incremental context updates
- **Custom Commands**: `@register_command` decorator for extensibility