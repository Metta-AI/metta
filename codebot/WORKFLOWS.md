# Workflow Composition Guide

This document explains how to compose commands into workflows for multi-step operations.

## Workflow Basics

Workflows combine multiple commands to accomplish complex tasks:
- Sequential execution with state passing
- Conditional branching based on results
- Parallel execution for independent operations
- Interactive handoff points

## Simple Workflow Example

```python
from codebot.workflow import Workflow, Step

# Test-debug-test loop
test_debug_workflow = Workflow(
    name="test-debug",
    description="Write tests, run them, debug failures",
    steps=[
        Step(
            name="write_tests",
            command="test",
            next_on_success="run_tests"
        ),
        Step(
            name="run_tests",
            command="pytest",  # Custom command to run tests
            next_on_success="complete",
            next_on_failure="debug_failures"
        ),
        Step(
            name="debug_failures",
            command="debug",
            next_on_success="run_tests"
        )
    ]
)
```

## State Management

Workflows maintain state between steps:

```python
@dataclass
class WorkflowState:
    initial_context: ExecutionContext
    completed_steps: List[str]
    step_outputs: Dict[str, CommandOutput]
    accumulated_changes: List[FileChange]
    
    def get_context_for_step(self, step_name: str) -> ExecutionContext:
        # Build context including previous step outputs
        context = self.initial_context.copy()
        context.add_metadata("previous_steps", self.step_outputs)
        return context
```

## Conditional Branching

```python
class ConditionalStep(Step):
    def __init__(self, name: str, condition: str, branches: Dict[str, str]):
        self.condition = condition  # Prompt to evaluate
        self.branches = branches     # outcome -> next_step mapping
    
    def get_next_step(self, output: CommandOutput) -> str:
        # Use LLM to evaluate condition based on output
        decision = self.evaluate_condition(output)
        return self.branches.get(decision, "default")
```

## Parallel Execution

```python
class ParallelStep(Step):
    def __init__(self, name: str, commands: List[str]):
        self.commands = commands
    
    async def execute(self, context: ExecutionContext) -> List[CommandOutput]:
        # Execute commands concurrently
        tasks = [
            execute_command(cmd, context) 
            for cmd in self.commands
        ]
        return await asyncio.gather(*tasks)
```

## Complex Workflow Example

```python
# Web application development workflow
webapp_workflow = Workflow(
    name="webapp",
    description="Design and implement a web application",
    steps=[
        # 1. Parallel design phase
        ParallelStep(
            name="design",
            commands=["design_ui", "design_api"]
        ),
        
        # 2. Review and refine designs
        Step(
            name="review_designs",
            command="review",
            custom_prompt="Review the UI mockups and API design for consistency"
        ),
        
        # 3. Conditional implementation
        ConditionalStep(
            name="check_complexity",
            condition="Is this a simple CRUD app or complex application?",
            branches={
                "simple": "implement_basic",
                "complex": "implement_modular"
            }
        ),
        
        # 4. Testing
        Step(
            name="test_integration",
            command="test",
            custom_context=["api/", "frontend/"]
        ),
        
        # 5. Interactive refinement
        InteractiveStep(
            name="polish",
            message="The basic implementation is complete. Please refine the UI/UX interactively."
        )
    ]
)
```

## Execution Engine

```python
class WorkflowEngine:
    async def execute(self, workflow: Workflow, initial_args: CommandArgs):
        state = WorkflowState(
            initial_context=self.build_initial_context(initial_args),
            completed_steps=[],
            step_outputs={},
            accumulated_changes=[]
        )
        
        current_step = workflow.get_first_step()
        
        while current_step:
            # Execute step with appropriate mode
            output = await self.execute_step(current_step, state)
            
            # Update state
            state.completed_steps.append(current_step.name)
            state.step_outputs[current_step.name] = output
            state.accumulated_changes.extend(output.file_changes)
            
            # Apply file changes
            for change in output.file_changes:
                self.apply_file_change(change)
            
            # Get next step
            current_step = current_step.get_next_step(output)
```

## Best Practices

### Workflow Design

1. **Keep Steps Focused**: Each step should have a clear purpose
2. **Minimize State**: Pass only necessary information between steps
3. **Handle Failures**: Always define failure paths
4. **Test Incrementally**: Validate output at each step

### State Propagation

```python
# Good: Explicit state passing
step_context = {
    "api_design": previous_outputs["design_api"],
    "ui_mockups": previous_outputs["design_ui"]
}

# Bad: Passing entire history
step_context = all_previous_outputs  # Too much context
```

### Error Recovery

```python
class RetryableStep(Step):
    max_retries: int = 3
    
    async def execute_with_retry(self, context: ExecutionContext):
        for attempt in range(self.max_retries):
            try:
                return await self.execute(context)
            except RecoverableError as e:
                if attempt == self.max_retries - 1:
                    raise
                context.add_error_context(str(e))
```

## Workflow Persistence

For long-running workflows:

```python
@dataclass
class WorkflowCheckpoint:
    workflow_id: str
    state: WorkflowState
    timestamp: datetime
    
    def save(self):
        # Persist to disk/database
        path = f".codebot/workflows/{self.workflow_id}.json"
        path.write_text(self.to_json())
    
    @classmethod
    def load(cls, workflow_id: str):
        path = f".codebot/workflows/{workflow_id}.json"
        return cls.from_json(path.read_text())
```

## CLI Integration

```bash
# Execute workflow
codebot workflow test-debug

# Execute with specific mode
codebot workflow webapp -p  # All steps in persistent mode

# Resume from checkpoint
codebot workflow --resume webapp

# List available workflows
codebot workflow --list
```