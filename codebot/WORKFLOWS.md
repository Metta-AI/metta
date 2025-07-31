# Workflows: Composing Multi-Step Development Tasks

Workflows chain commands together to accomplish complex development tasks with conditional logic, parallel execution, and intelligent orchestration.

## Quick Examples

```bash
# Run a test-driven development workflow
codebot tdd src/feature.py

# Complete feature development workflow
codebot feature -m "Add user notifications"

# Automated PR preparation
codebot prepare-pr

# Custom workflow
codebot @workflow "modernize-codebase" src/
```

## Core Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

class StepResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIP = "skip"
    RETRY = "retry"

class WorkflowStep(BaseModel):
    """Single step in a workflow"""
    id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    name: str
    command: Command  # Reuses command from CODEBOT.md

    # Conditional execution
    condition: Optional[str] = None  # Python expression
    on_success: Optional[str] = None  # Next step ID
    on_failure: Optional[str] = None  # Next step ID

    # Retry logic
    max_retries: int = 3
    retry_delay: float = 1.0

    # Parallel execution
    parallel_with: List[str] = []  # Step IDs to run in parallel

    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if step should execute based on condition"""
        if not self.condition:
            return True

        # Safe evaluation of condition
        try:
            return eval(self.condition, {"__builtins__": {}}, context)
        except:
            return False

class Workflow(BaseModel):
    """Complete workflow definition"""
    name: str
    description: str
    steps: Dict[str, WorkflowStep]  # id -> step
    entry_point: str  # First step ID

    # Global settings
    max_parallel: int = 5
    stop_on_failure: bool = True
    timeout_seconds: Optional[int] = 3600

    def get_next_steps(self,
                      current_step: str,
                      result: StepResult) -> List[str]:
        """Determine next steps based on current result"""
        step = self.steps[current_step]

        if result == StepResult.SUCCESS and step.on_success:
            return [step.on_success]
        elif result == StepResult.FAILURE and step.on_failure:
            return [step.on_failure]

        # Find steps that depend on this one
        next_steps = []
        for sid, s in self.steps.items():
            if s.condition and f"steps['{current_step}'].success" in s.condition:
                next_steps.append(sid)

        return next_steps
```

## Workflow Execution Engine

```python
from pydantic_ai import Agent
import asyncio
from typing import Set

class WorkflowExecutor:
    """Executes workflows with parallel and conditional support"""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.step_results: Dict[str, StepResult] = {}
        self.step_outputs: Dict[str, CommandOutput] = {}
        self.execution_order: List[str] = []

        # PydanticAI agent for intelligent error handling
        self.error_handler = Agent(
            result_type=ErrorRecoveryStrategy,
            system_prompt="""Analyze workflow errors and suggest recovery strategies.
            Consider: retry with modifications, skip step, or fail workflow."""
        )

    async def execute(self, initial_context: ExecutionContext) -> WorkflowResult:
        """Execute complete workflow"""

        context = initial_context
        pending_steps: Set[str] = {self.workflow.entry_point}
        executing_steps: Set[str] = set()
        completed_steps: Set[str] = set()

        while pending_steps or executing_steps:
            # Start new steps up to parallel limit
            can_start = min(
                self.workflow.max_parallel - len(executing_steps),
                len(pending_steps)
            )

            if can_start > 0:
                # Find steps ready to execute
                ready_steps = []
                for step_id in list(pending_steps):
                    step = self.workflow.steps[step_id]

                    # Check if dependencies are met
                    if self._dependencies_met(step, completed_steps):
                        ready_steps.append(step_id)
                        if len(ready_steps) >= can_start:
                            break

                # Start execution
                for step_id in ready_steps:
                    pending_steps.remove(step_id)
                    executing_steps.add(step_id)
                    asyncio.create_task(self._execute_step(step_id, context))

            # Wait for at least one step to complete
            await asyncio.sleep(0.1)

            # Check completed steps
            for step_id in list(executing_steps):
                if step_id in self.step_results:
                    executing_steps.remove(step_id)
                    completed_steps.add(step_id)

                    # Add next steps
                    result = self.step_results[step_id]
                    next_steps = self.workflow.get_next_steps(step_id, result)
                    pending_steps.update(next_steps)

                    # Update context with step output
                    if step_id in self.step_outputs:
                        context = self._update_context(context, self.step_outputs[step_id])

            # Check for workflow failure
            if self.workflow.stop_on_failure:
                if any(r == StepResult.FAILURE for r in self.step_results.values()):
                    break

        return self._build_result()

    async def _execute_step(self,
                          step_id: str,
                          context: ExecutionContext) -> None:
        """Execute a single workflow step with retries"""

        step = self.workflow.steps[step_id]
        self.execution_order.append(step_id)

        # Check condition
        if not step.should_execute({"context": context, "steps": self.step_results}):
            self.step_results[step_id] = StepResult.SKIP
            return

        # Execute with retries
        for attempt in range(step.max_retries):
            try:
                output = await step.command.execute(context)
                self.step_outputs[step_id] = output
                self.step_results[step_id] = StepResult.SUCCESS
                return

            except Exception as e:
                if attempt < step.max_retries - 1:
                    # Ask AI for recovery strategy
                    strategy = await self._get_recovery_strategy(step, e, context)

                    if strategy.action == "retry_with_modification":
                        # Modify context based on AI suggestion
                        context = self._apply_recovery_modification(context, strategy)
                        await asyncio.sleep(step.retry_delay * (2 ** attempt))
                    elif strategy.action == "skip":
                        self.step_results[step_id] = StepResult.SKIP
                        return
                    else:
                        self.step_results[step_id] = StepResult.FAILURE
                        return
                else:
                    self.step_results[step_id] = StepResult.FAILURE

    async def _get_recovery_strategy(self,
                                   step: WorkflowStep,
                                   error: Exception,
                                   context: ExecutionContext) -> ErrorRecoveryStrategy:
        """Use AI to determine recovery strategy"""

        result = await self.error_handler.run({
            "step": step.model_dump(),
            "error": str(error),
            "error_type": type(error).__name__,
            "context_summary": self._summarize_context(context),
            "previous_attempts": self.execution_order.count(step.id)
        })

        return result.data
```

## Built-in Workflows

### Test-Driven Development (TDD)

```python
TDD_WORKFLOW = Workflow(
    name="tdd",
    description="Test-driven development workflow",
    steps={
        "write_test": WorkflowStep(
            name="Write failing test",
            command=Command(
                name="test",
                prompt_template="""Write a comprehensive test for the feature.
                The test should fail initially since the feature isn't implemented."""
            ),
            on_success="run_test",
            on_failure="clarify_spec"
        ),

        "run_test": WorkflowStep(
            name="Run test to verify it fails",
            command=Command(
                name="run-tests",
                prompt_template="Run the test and verify it fails as expected"
            ),
            on_success="implement",
            on_failure="fix_test"
        ),

        "implement": WorkflowStep(
            name="Implement feature",
            command=Command(
                name="implement",
                prompt_template="""Implement the minimal code to make the test pass.
                Focus on making the test green, not on perfect code."""
            ),
            on_success="verify_pass",
            on_failure="debug"
        ),

        "verify_pass": WorkflowStep(
            name="Verify tests pass",
            command=Command(
                name="run-tests",
                prompt_template="Run tests to verify they now pass"
            ),
            on_success="refactor",
            on_failure="debug"
        ),

        "refactor": WorkflowStep(
            name="Refactor implementation",
            command=Command(
                name="refactor",
                prompt_template="""Refactor the code while keeping tests green.
                Improve structure, remove duplication, enhance readability."""
            ),
            on_success="final_test"
        ),

        "clarify_spec": WorkflowStep(
            name="Clarify requirements",
            command=Command(
                name="clarify",
                prompt_template="Analyze why test writing failed and clarify requirements"
            ),
            on_success="write_test"
        )
    },
    entry_point="write_test"
)
```

### Feature Development

```python
def create_feature_workflow(feature_description: str) -> Workflow:
    """Create a complete feature development workflow"""

    return Workflow(
        name="feature",
        description=f"Develop feature: {feature_description}",
        steps={
            "analyze": WorkflowStep(
                name="Analyze requirements",
                command=Command(
                    name="analyze",
                    prompt_template=f"""Analyze the requirements for: {feature_description}
                    Identify: affected files, necessary changes, potential challenges"""
                ),
                on_success="design"
            ),

            "design": WorkflowStep(
                name="Design solution",
                command=Command(
                    name="design",
                    prompt_template="Create a design for the feature implementation"
                ),
                on_success="write_tests"
            ),

            "write_tests": WorkflowStep(
                name="Write tests",
                command=COMMANDS["test"],
                on_success="implement_core",
                parallel_with=["update_docs"]
            ),

            "implement_core": WorkflowStep(
                name="Implement core functionality",
                command=COMMANDS["implement"],
                on_success="implement_edge",
                max_retries=5
            ),

            "implement_edge": WorkflowStep(
                name="Handle edge cases",
                command=Command(
                    name="implement-edges",
                    prompt_template="Add error handling and edge case handling"
                ),
                on_success="integration_test"
            ),

            "update_docs": WorkflowStep(
                name="Update documentation",
                command=Command(
                    name="document",
                    prompt_template="Update documentation for the new feature"
                )
            ),

            "integration_test": WorkflowStep(
                name="Integration testing",
                command=Command(
                    name="test-integration",
                    prompt_template="Write and run integration tests"
                ),
                on_success="review"
            ),

            "review": WorkflowStep(
                name="Final review",
                command=COMMANDS["review"],
                condition="all(steps[s].success for s in ['implement_core', 'implement_edge', 'update_docs'])"
            )
        },
        entry_point="analyze",
        max_parallel=3
    )
```

## Custom Workflow Definition

### Using Python

```python
# ~/.codebot/workflows.py

from codebot import Workflow, WorkflowStep, register_workflow

@register_workflow
def database_migration_workflow():
    """Workflow for safe database migrations"""

    return Workflow(
        name="db-migration",
        description="Safely apply database migrations",
        steps={
            "backup": WorkflowStep(
                name="Backup database",
                command=Command(
                    name="backup-db",
                    prompt_template="Create database backup script"
                ),
                on_failure="abort"
            ),

            "analyze": WorkflowStep(
                name="Analyze migration impact",
                command=Command(
                    name="analyze-migration",
                    prompt_template="""Analyze the migration for:
                    - Data loss risks
                    - Performance impact
                    - Rollback strategy"""
                ),
                on_success="generate"
            ),

            "generate": WorkflowStep(
                name="Generate migration",
                command=Command(
                    name="generate-migration",
                    prompt_template="Generate migration with rollback support"
                ),
                on_success="test",
                parallel_with=["update_models"]
            ),

            "test": WorkflowStep(
                name="Test migration",
                command=Command(
                    name="test-migration",
                    prompt_template="Test migration on copy of production data"
                ),
                on_success="apply",
                on_failure="rollback_plan"
            ),

            "apply": WorkflowStep(
                name="Apply migration",
                command=Command(
                    name="apply-migration",
                    prompt_template="Apply migration with monitoring"
                ),
                condition="steps['test'].success and user_approval"
            )
        },
        entry_point="backup",
        stop_on_failure=True
    )
```

### Using YAML

```yaml
# ~/.codebot/workflows/code-review.yaml

name: comprehensive-review
description: Thorough code review workflow

steps:
  static_analysis:
    name: Run static analysis
    command: lint
    parallel_with: [security_scan, type_check]

  security_scan:
    name: Security vulnerability scan
    command: security-audit

  type_check:
    name: Type checking
    command: type-check

  test_coverage:
    name: Check test coverage
    command: coverage
    condition: all_parallel_success
    on_failure: write_tests

  write_tests:
    name: Write missing tests
    command: test
    on_success: test_coverage

  performance:
    name: Performance analysis
    command: profile
    condition: steps['test_coverage'].success

  final_review:
    name: AI code review
    command: review
    condition: all(steps[s].success for s in previous_steps)

entry_point: static_analysis
max_parallel: 3
```

## Unified Execution in Workflow Graphs

### Leaf Node Execution 

In workflow graphs, leaf nodes (AI agents) need to work across different execution modes while receiving the same inputs:

```python
from codebot import execute_agent

class WorkflowStep(BaseModel):
    """Enhanced workflow step with execution mode support"""
    id: str
    name: str
    command: Command
    execution_mode: Optional[str] = None  # 'remote', 'pipeline', 'interactive'
    
    # ... existing fields ...
    
    async def execute(self, context: ExecutionContext) -> CommandOutput:
        """Execute step with mode override"""
        
        # Use workflow-level mode or step-specific mode
        mode = self.execution_mode or context.get('_workflow_mode')
        
        # Same command, different execution
        return await self.command.execute(context, mode=mode)

class Workflow(BaseModel):
    """Workflow with execution mode support"""
    name: str
    description: str
    steps: Dict[str, WorkflowStep]
    entry_point: str
    execution_mode: Optional[str] = None  # Apply to all steps
    
    async def execute_with_mode(self, 
                               initial_context: ExecutionContext,
                               mode: Optional[str] = None) -> WorkflowResult:
        """Execute workflow in specific mode"""
        
        # Set mode for all steps
        if mode or self.execution_mode:
            initial_context['_workflow_mode'] = mode or self.execution_mode
        
        # If pipeline mode, create shared session
        if initial_context.get('_workflow_mode') == 'pipeline':
            initial_context['_pipeline_session_id'] = f"workflow_{self.name}"
        
        # Execute normally - steps will use the mode
        executor = WorkflowExecutor(self)
        return await executor.execute(initial_context)

# Example: Same workflow, different modes
async def run_tdd_workflow(mode: str = 'remote'):
    """Run TDD workflow in specified mode"""
    
    workflow = TDD_WORKFLOW  # From earlier definition
    context = ExecutionContext()
    
    if mode == 'interactive':
        # Will launch Claude Code for each step
        print("Running TDD workflow with Claude Code...")
        # Note: Interactive mode won't return structured results
        try:
            await workflow.execute_with_mode(context, mode='interactive')
        except NotImplementedError:
            print("Claude Code launched for manual execution")
    
    elif mode == 'pipeline':
        # Stateful conversation through workflow
        print("Running TDD workflow in pipeline mode...")
        # Note: Pipeline mode structured output is still being developed
        result = await workflow.execute_with_mode(context, mode='pipeline')
        return result
    
    else:
        # Standard remote execution
        result = await workflow.execute_with_mode(context, mode='remote')
        return result

# Mixed-mode workflow
mixed_workflow = Workflow(
    name="mixed_mode_review",
    description="Code review with mixed execution",
    steps={
        "analyze": WorkflowStep(
            name="Analyze code",
            command=COMMANDS["analyze"],
            execution_mode="remote"  # Always use remote for speed
        ),
        "review": WorkflowStep(
            name="Review findings",
            command=COMMANDS["review"],
            execution_mode="interactive"  # Hand off to human
        ),
        "fix": WorkflowStep(
            name="Apply fixes",
            command=COMMANDS["fix"],
            execution_mode="pipeline"  # Stateful conversation
        )
    },
    entry_point="analyze"
)
```

### Simplified Usage

```python
# The key insight: same inputs, different execution
async def demo_execution_modes():
    # Define the task
    command = Command(
        name="refactor",
        prompt_template="Refactor this code for clarity",
        default_paths=["src/*.py"]
    )
    
    context = ExecutionContext(
        files={"src/main.py": "def f(x,y): return x+y"},
        git_diff="",
        clipboard=""
    )
    
    # Execute in different modes - same inputs!
    
    # Remote: Returns structured output
    result = await command.execute(context, mode="remote")
    print(f"Remote result: {result.file_changes}")
    
    # Pipeline: Stateful conversation, returns structured output
    result = await command.execute(context, mode="pipeline")
    print(f"Pipeline result: {result.file_changes}")
    
    # Interactive: Launches Claude Code
    try:
        await command.execute(context, mode="interactive")
    except NotImplementedError:
        print("Claude Code launched with context")
```

## Advanced Patterns

### Dynamic Workflow Generation

```python
class DynamicWorkflowBuilder:
    """Build workflows based on project analysis"""

    def __init__(self):
        self.analyzer = Agent(
            result_type=WorkflowBlueprint,
            system_prompt="""Analyze the project and create an optimal workflow.
            Consider: project type, existing tools, team preferences."""
        )

    async def build_workflow(self,
                           goal: str,
                           project_context: Dict[str, Any]) -> Workflow:
        """Generate workflow dynamically based on project needs"""

        # Analyze project and determine optimal workflow
        blueprint = await self.analyzer.run({
            "goal": goal,
            "project_type": project_context.get("type", "unknown"),
            "available_tools": self._detect_tools(),
            "file_structure": self._analyze_structure(),
            "team_preferences": project_context.get("preferences", {})
        })

        # Convert blueprint to workflow
        return self._blueprint_to_workflow(blueprint.data)
```

### Workflow Composition

```python
class WorkflowComposer:
    """Combine smaller workflows into larger ones"""

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
            # Add all steps with unique IDs
            for step_id, step in workflow.steps.items():
                new_id = f"{workflow.name}_{step_id}"
                combined.steps[new_id] = step.model_copy()

                # Connect workflows
                if last_exit and step_id == workflow.entry_point:
                    combined.steps[last_exit].on_success = new_id

            if not combined.entry_point:
                combined.entry_point = f"{workflow.name}_{workflow.entry_point}"

            # Find exit points
            last_exit = self._find_exit_point(workflow, workflow.name)

        return combined

    def parallel(self, *workflows: Workflow) -> Workflow:
        """Run workflows in parallel"""
        # Implementation...
```

## Integration with Other Modes

### Workflows in Loops

```python
class WorkflowLoop(Loop):
    """Continuously execute workflows toward a goal"""

    def __init__(self, workflow: Workflow, goal: str):
        self.workflow = workflow
        self.executor = WorkflowExecutor(workflow)
        super().__init__(
            command=Command(
                name=f"workflow_{workflow.name}",
                prompt_template=workflow.description
            ),
            goal=goal
        )

    async def run_iteration(self) -> bool:
        """Run workflow as loop iteration"""

        # Execute workflow
        result = await self.executor.execute(self.context)

        # Update metrics based on workflow result
        self.state.metrics.update({
            "workflow_success_rate": result.success_rate,
            "steps_completed": len(result.completed_steps),
            "time_taken": result.duration_seconds
        })

        # Continue if goal not met
        return not self._goal_achieved()
```

### Workflows with Manybots

```python
# Manybots can execute workflows as part of their work cycles
class WorkflowManybot(Manybot):
    """Manybot that executes workflows to achieve goals"""

    def __init__(self, name: str, goal: Goal, workflow: Workflow):
        super().__init__(name, goal, aors=[])
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor(workflow)

    async def do_work(self) -> WorkCycle:
        """Execute workflow as work cycle"""

        # Run workflow
        result = await self.workflow_executor.execute(self.context)

        # Convert to work cycle
        return WorkCycle(
            cycle_id=str(uuid.uuid4()),
            bot_id=self.state.bot_id,
            timestamp=datetime.now(),
            goal_evaluation=self.goal.evaluate(),
            planned_tasks=[step.name for step in self.workflow.steps.values()],
            executed_commands=[s for s in result.completed_steps],
            changes_made=self._extract_changes(result)
        )
```
