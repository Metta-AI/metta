# Codebot Design Overview

## Architecture

### Three Execution Modes

Codebot supports three LLM interaction modes, each optimized for different use cases:

1. **One-shot Mode (default)**: Direct LLM API call
   - Fastest performance
   - No conversation state
   - Best for simple, well-defined tasks
   
2. **Persistent Mode (-p)**: Uses `claude -p`
   - Maintains conversation context
   - Allows iterative refinement
   - Good for complex tasks requiring multiple steps

3. **Interactive Mode (-i)**: Launches Claude Code
   - Full IDE integration
   - Human-in-the-loop refinement
   - Best for exploratory or creative tasks

### Core Components

```python
# Command Definition
@dataclass
class Command:
    name: str
    description: str
    prompt_template: str
    default_paths: List[str]  # e.g., ["tests/"] for test command
    output_schema: OutputSchema

# Execution Context
@dataclass
class ExecutionContext:
    git_diff: str              # Current uncommitted changes
    clipboard: str             # User's clipboard content
    relevant_files: List[FileContent]  # Selected based on task
    working_directory: Path
    mode: ExecutionMode

# Command Output
@dataclass
class CommandOutput:
    file_changes: List[FileChange]
    summary: str
    metadata: Dict[str, Any]  # Mode-specific data

@dataclass
class FileChange:
    filepath: str
    content: str
    operation: Literal["write", "delete"]
```

### Execution Flow

```python
class CommandExecutor:
    def execute(self, command: Command, args: CommandArgs) -> CommandOutput:
        # 1. Gather context
        context = self.context_provider.gather(command, args)
        
        # 2. Route to appropriate executor
        if args.interactive:
            return self.execute_interactive(command, context)
        elif args.persistent:
            return self.execute_persistent(command, context)
        else:
            return self.execute_oneshot(command, context)
    
    def execute_oneshot(self, command: Command, context: ExecutionContext) -> CommandOutput:
        # Direct LLM API call
        prompt = self.build_prompt(command.prompt_template, context)
        response = self.llm_client.complete(prompt)
        return self.parse_output(response)
    
    def execute_persistent(self, command: Command, context: ExecutionContext) -> CommandOutput:
        # Use claude -p for stateful execution
        prompt = self.build_prompt(command.prompt_template, context)
        result = subprocess.run(["claude", "-p", prompt], capture_output=True)
        return self.parse_output(result.stdout)
    
    def execute_interactive(self, command: Command, context: ExecutionContext) -> CommandOutput:
        # Launch Claude Code with prepared context
        subprocess.run(["claude", f"Please help with {command.description}. Context: {context.summary}"])
        return CommandOutput(file_changes=[], summary="Launched interactive session")
```

## Context Management

### Smart File Selection

```python
class ContextProvider:
    def gather(self, command: Command, args: CommandArgs) -> ExecutionContext:
        # Always include
        git_diff = self.get_git_diff()
        clipboard = self.get_clipboard()
        
        # Command-specific defaults
        default_files = self.get_files(command.default_paths)
        
        # User-specified paths
        user_files = self.get_files(args.paths) if args.paths else []
        
        # Smart selection within token budget
        relevant_files = self.select_relevant_files(
            default_files + user_files,
            max_tokens=args.max_tokens or 50000
        )
        
        return ExecutionContext(
            git_diff=git_diff,
            clipboard=clipboard,
            relevant_files=relevant_files,
            working_directory=Path.cwd(),
            mode=args.mode
        )
```

### Token Budgeting

```python
def select_relevant_files(self, files: List[FileContent], max_tokens: int) -> List[FileContent]:
    # Prioritize by:
    # 1. Files mentioned in git diff
    # 2. Test files for test commands
    # 3. Import relationships
    # 4. Recency of modification
    # 5. File size (prefer smaller files)
    
    selected = []
    current_tokens = 0
    
    for file in self.prioritize_files(files):
        file_tokens = self.count_tokens(file.content)
        if current_tokens + file_tokens <= max_tokens:
            selected.append(file)
            current_tokens += file_tokens
    
    return selected
```

## Command Examples

### Test Command

```python
test_command = Command(
    name="test",
    description="Write comprehensive tests for changed code",
    prompt_template="""
You are a test writing expert. Analyze the provided code changes and write comprehensive tests.

Guidelines:
- Write tests that verify behavior, not implementation
- Cover edge cases and error conditions
- Use existing test patterns from the codebase
- Ensure tests are maintainable and clear

Context:
{context}

Generate test files that thoroughly test the changed code.
""",
    default_paths=["tests/", "**/test_*.py", "**/*_test.py"],
    output_schema=OutputSchema(
        file_patterns=["test_*.py", "*_test.py"],
        required_fields=["file_changes"]
    )
)
```

### Review Command

```python
review_command = Command(
    name="review",
    description="Review code changes for quality and security",
    prompt_template="""
You are a senior engineer conducting a code review. Analyze the changes for:
- Correctness and logic errors
- Security vulnerabilities
- Performance issues
- Code style and maintainability

Context:
{context}

Provide a review.md file with detailed feedback organized by severity.
""",
    default_paths=[".github/PULL_REQUEST_TEMPLATE.md", "CONTRIBUTING.md"],
    output_schema=OutputSchema(
        file_patterns=["review.md"],
        required_fields=["file_changes", "summary"]
    )
)
```

## Workflows

### Simple Sequential Workflow

```python
@dataclass
class Workflow:
    name: str
    steps: List[WorkflowStep]

@dataclass
class WorkflowStep:
    command: Command
    on_success: Optional[str]  # Next step name
    on_failure: Optional[str]  # Alternative step name

# Example: test-debug workflow
test_debug_workflow = Workflow(
    name="test-debug",
    steps=[
        WorkflowStep(
            command=test_command,
            on_success="check_tests",
            on_failure=None
        ),
        WorkflowStep(
            name="check_tests",
            command=run_tests_command,
            on_success="done",
            on_failure="debug"
        ),
        WorkflowStep(
            name="debug",
            command=debug_command,
            on_success="check_tests",
            on_failure=None
        )
    ]
)
```

### Workflow Execution

```python
class WorkflowExecutor:
    def execute(self, workflow: Workflow, initial_context: ExecutionContext):
        current_step = workflow.steps[0]
        context = initial_context
        
        while current_step:
            # Execute command
            output = self.command_executor.execute(current_step.command, context)
            
            # Apply file changes
            for change in output.file_changes:
                self.apply_file_change(change)
            
            # Update context for next step
            context = self.update_context(context, output)
            
            # Determine next step
            if output.success and current_step.on_success:
                current_step = self.find_step(workflow, current_step.on_success)
            elif not output.success and current_step.on_failure:
                current_step = self.find_step(workflow, current_step.on_failure)
            else:
                break
```

## Prompt Management

### Component-Based Prompts

```python
class PromptBuilder:
    def __init__(self, components_dir: Path):
        self.components = self.load_components(components_dir)
    
    def build_prompt(self, template: str, components: List[str]) -> str:
        # Load and combine components
        sections = [self.components[name] for name in components]
        
        # Build final prompt
        return template.format(
            components="\n\n".join(sections)
        )

# Example components structure:
# prompts/
# ├── components/
# │   ├── testing_guidelines.md
# │   ├── security_checklist.md
# │   ├── code_style.md
# │   └── performance_tips.md
# └── commands/
#     ├── test.md
#     └── review.md
```

## Remote Execution (Future)

```python
# Subscription to repository events
@dataclass
class Subscription:
    id: str
    repo: str
    paths: List[str]
    workflow: str
    trigger: Literal["push", "pr", "schedule"]

# Example usage:
# remotebot subscribe test src/ --trigger=push
# remotebot list
# remotebot logs <subscription-id>
```

## Implementation Guidelines

### Command Implementation Checklist

1. Define clear prompt template
2. Specify default paths for context
3. Create output schema for validation
4. Handle all three execution modes
5. Write tests for command behavior
6. Document usage examples

### Best Practices

1. **Keep Commands Focused**: Each command should do one thing well
2. **Optimize Context**: Include only relevant files within token limits
3. **Clear Output**: Structured file changes with clear operations
4. **Error Handling**: Graceful failures with helpful messages
5. **Mode Selection**: Choose the right mode for the task

### Performance Considerations

- One-shot mode: ~2-5 seconds for simple tasks
- Persistent mode: ~5-10 seconds with context retention
- Interactive mode: Human-speed interaction
- Context gathering: Optimize file selection algorithms
- Token usage: Monitor and budget appropriately