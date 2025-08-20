# Goalbot

```bash
# Define a goal and let the agent work until completion
goalbot achieve "Add comprehensive error handling to the API" \
  --context src/api \
  --success "All endpoints have try-catch blocks and return proper error codes"

# Resume a previous goal session
goalbot resume goal-123

# Check goal progress
goalbot status goal-123
```

## Goal-Driven Development

Goalbot takes a high-level goal and works iteratively until success criteria are met:

1. **Receive Goal**: Accept goal with context and success criteria
2. **Break into Tasks**: Decompose goal into concrete tasks
3. **Execute Loop**: Use `claude -p` to maintain context across iterations
4. **Evaluate Progress**: Check success criteria after each iteration
5. **Continue or Complete**: Loop until success or unable to progress

## Service Interface

### Provides
- Goal planning and task breakdown
- Claude SDK integration for autonomous task execution
- Success criteria evaluation
- Progress tracking and checkpointing

### Consumes
- codebot: For all file operations and code modifications

## Core Concepts

```python
class Goal(BaseModel):
    """High-level objective with success criteria"""
    description: str  # What needs to be achieved
    context_paths: List[str]  # Files/directories relevant to goal
    success_criteria: str  # Single, clear success criterion

class Task(BaseModel):
    """Individual task within a goal"""
    description: str
    command: str  # Codebot command or workflow name
    priority: int = 1
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    dependencies: List[str] = Field(default_factory=list)  # Other task IDs
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are met"""
        return all(dep in completed_tasks for dep in self.dependencies)

class GoalSession(BaseModel):
    """Persistent goal execution state"""
    goal_id: str
    goal: Goal
    tasks: List[Task] = Field(default_factory=list)
    completed_tasks: List[Task] = Field(default_factory=list)
    context_checkpoints: List[ContextSnapshot] = Field(default_factory=list)
    iteration_count: int = 0
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    async def continue_work(self) -> TaskResult:
        """Resume from last checkpoint"""
        # Each claude -p execution runs to completion autonomously
        
        # Get next ready task
        completed_ids = {t.description for t in self.completed_tasks}
        next_task = next(
            (t for t in self.tasks 
             if t.status == "pending" and t.is_ready(completed_ids)),
            None
        )
        
        if not next_task:
            # No ready tasks, need to replan
            return None
        
        # Execute via codebot command or workflow
        result = await codebot.execute(
            command=next_task.command,
            context=self._build_context(),
            mode="claudesdk"  # Autonomous execution via claude -p
        )
        
        # Update task status
        next_task.status = "completed" if result.success else "failed"
        
        # Checkpoint progress
        self.context_checkpoints.append(
            ContextSnapshot(
                task=next_task,
                result=result,
                timestamp=datetime.now()
            )
        )
        return result

class SuccessEvaluation(BaseModel):
    """Result of evaluating goal success"""
    criteria_met: bool
    evidence: List[str]
    missing_requirements: List[str]
    confidence: float  # 0.0 to 1.0
```

## Goal Execution Workflow

```python
class GoalExecutor:
    """Executes goals through iterative task completion"""
    
    def __init__(self, goal: Goal):
        self.goal = goal
        self.session = GoalSession(
            goal=goal,
            tasks=[]
        )
        
        # Import codebot for command execution
        from codebot import Command, ContextManager
        self.codebot = Command(default_mode="claudesdk")  # Goalbot uses claudesdk
        self.context_manager = ContextManager()
    
    async def achieve(self) -> GoalResult:
        """Main goal achievement loop"""
        
        # 1. Initial task breakdown
        self.session.tasks = await self._break_into_tasks()
        
        # 2. Execute loop
        while self._should_continue():
            # Get next task
            task = self._get_next_task()
            if not task:
                # No more tasks, check if we need more
                if await self._evaluate_success():
                    return GoalResult(success=True, session=self.session)
                
                # Generate more tasks
                new_tasks = await self._generate_additional_tasks()
                if not new_tasks:
                    return GoalResult(
                        success=False, 
                        reason="Unable to generate more tasks",
                        session=self.session
                    )
                self.session.tasks.extend(new_tasks)
                continue
            
            # Execute task via Claude SDK
            result = await self._execute_task(task)
            
            # Update session
            task.status = TaskStatus.COMPLETED
            self.session.completed_tasks.append(task)
            self.session.iteration_count += 1
            
            # Add result to context for next iteration
            self.session.context_history.append({
                "task": task.description,
                "result": result.summary,
                "files_changed": [fc.filepath for fc in result.file_changes]
            })
        
        return GoalResult(
            success=await self._evaluate_success(),
            session=self.session
        )
    
    async def _break_into_tasks(self) -> List[Task]:
        """Use AI to break goal into tasks"""
        from pydantic_ai import Agent
        
        # Define task planning result structure
        class TaskPlan(BaseModel):
            tasks: List[Task]
            rationale: str
            estimated_iterations: int
        
        task_planner = Agent(
            result_type=TaskPlan,
            system_prompt="""Break down the goal into concrete, actionable tasks.
            Each task should be achievable with a single codebot command or workflow.
            Consider dependencies and order tasks logically.
            Identify which tasks can be done in parallel."""
        )
        
        # Analyze codebase for context
        codebase_info = await self._analyze_codebase()
        
        context = {
            "goal": self.goal.description,
            "success_criteria": self.goal.success_criteria,
            "codebase_analysis": codebase_info,
            "available_commands": list(CODEBOT_COMMANDS.keys()),
            "available_workflows": ["tdd", "feature", "refactor-suite"]
        }
        
        result = await task_planner.run(context)
        plan = result.data
        
        # Log planning rationale
        self.session.metadata["plan_rationale"] = plan.rationale
        self.session.metadata["estimated_iterations"] = plan.estimated_iterations
        
        return plan.tasks
    
    async def _execute_task(self, task: Task) -> CommandOutput:
        """Execute task using claude -p for context continuity"""
        
        # Build cumulative context
        full_context = {
            "goal": self.goal.description,
            "success_criteria": self.goal.success_criteria,
            "current_task": task.description,
            "completed_tasks": [t.description for t in self.session.completed_tasks],
            "remaining_tasks": [t.description for t in self.session.tasks if t.status == TaskStatus.PENDING],
            "previous_results": self.session.context_history[-5:]  # Last 5 results
        }
        
        # Execute via Claude SDK for autonomous completion
        return await self.codebot.execute(
            task.command,
            context=full_context,
            mode="claudesdk"
        )
    
    async def _evaluate_success(self) -> bool:
        """Check if success criteria are met"""
        from pydantic_ai import Agent
        
        evaluator = Agent(
            result_type=SuccessEvaluation,
            system_prompt="Evaluate if the success criteria have been met"
        )
        
        context = {
            "goal": self.goal.description,
            "success_criteria": self.goal.success_criteria,
            "completed_tasks": [t.description for t in self.session.completed_tasks],
            "current_state": await self._analyze_current_state()
        }
        
        result = await evaluator.run(context)
        return result.data.criteria_met
    
    def _should_continue(self) -> bool:
        """Determine if we should continue working"""
        MAX_ITERATIONS = 50
        
        # Stop conditions
        if self.session.iteration_count >= MAX_ITERATIONS:
            return False
        
        # Continue if we have pending tasks or can generate more
        return True
```

## CLI Implementation

```python
@click.command()
@click.argument('goal_description')
@click.option('--context', '-c', multiple=True, help='Context paths')
@click.option('--success', '-s', required=True, help='Success criteria')
@click.option('--resume', help='Resume from session ID')
def achieve(goal_description: str, context: tuple, success: str, resume: str):
    """Achieve a goal through iterative task execution"""
    
    async def run():
        if resume:
            # Load existing session
            session = await load_session(resume)
            executor = GoalExecutor.from_session(session)
        else:
            # Create new goal
            goal = Goal(
                description=goal_description,
                context_paths=list(context) or ['.'],
                success_criteria=success
            )
            executor = GoalExecutor(goal)
        
        # Run until completion
        print(f"Working on goal: {executor.goal.description}")
        print(f"Success criteria: {executor.goal.success_criteria}")
        print("Starting autonomous execution...\n")
        
        result = await executor.achieve()
        
        if result.success:
            print(f"\n✓ Goal achieved in {result.session.iteration_count} iterations!")
        else:
            print(f"\n✗ Unable to achieve goal: {result.reason}")
            print(f"Completed {len(result.session.completed_tasks)} tasks")
            print(f"Session ID: {result.session.goal.id}")
            print("Run with --resume to continue")
    
    asyncio.run(run())
```

## Example Usage

```bash
# Add error handling
goalbot achieve "Add comprehensive error handling to the API" \
  --context src/api \
  --success "All endpoints have try-catch and return proper error codes"

# Improve test coverage  
goalbot achieve "Increase test coverage to 90%" \
  --context src tests \
  --success "Coverage report shows >= 90%"
```

## Claude SDK Execution Example

```python
class ClaudeSDKContext:
    """Manages Claude SDK execution for goalbot iterations"""
    
    def __init__(self, goal: Goal):
        self.goal = goal
        self.execution_history = []
        
    async def execute_with_claudesdk(self, task: Task) -> CommandOutput:
        """Execute task using claude -p (autonomous completion)"""
        
        # Build context including history
        context = ExecutionContext(
            git_diff=await self._get_current_diff(),
            files=await self._gather_relevant_files(),
            metadata={
                "goal": self.goal.description,
                "completed_tasks": [t.description for t in self.completed_tasks],
                "current_task": task.description
            }
        )
        
        # Execute via codebot in Claude SDK mode
        result = await codebot.execute(
            command=task.command,
            context=context,
            mode="claudesdk"
        )
        
        # Log the output for debugging
        logger.info(f"Claude SDK completed task: {task.description}")
        
        # Update conversation history
        self.conversation_history.append({
            "task": task,
            "result": result,
            "timestamp": datetime.now()
        })
        
        return result

# Usage showing iterative Claude SDK execution
context = ClaudeSDKContext(goal)

# First task - establishes context
result1 = await context.execute_with_memory(
    Task(description="Write failing test", command="test")
)

# Second task - has access to previous context
result2 = await context.execute_with_memory(
    Task(description="Implement to pass test", command="implement")
)
# Claude remembers the test from task 1!
```

## CLI Usage

```bash
# Start new goal
goalbot achieve "Refactor authentication to use JWT tokens" \
  --context src/auth \
  --success "All auth endpoints use JWT instead of sessions"

# Monitor progress
goalbot status goal_abc123

# Resume after interruption  
goalbot resume goal_abc123

# List all goals
goalbot list
goalbot list --status active

# Show detailed log
goalbot log goal_abc123
```

## Integration Points

- **Codebot**: Uses codebot commands for all file operations
- **Remotebot**: Can be deployed as a remote agent via remotebot
- **Manybot**: Goals can be assigned to manybots as their objectives

## Architecture Principles

1. **Goal-Oriented**: Everything starts with a clear goal and success criteria
2. **Iterative Progress**: Work in small, verifiable steps
3. **Iterative Execution**: Each task runs to completion via claude -p
4. **Human-in-the-Loop**: Submit PRs for human review before applying changes
5. **Resumable**: Can pause and resume goal achievement

## Implementation Roadmap

### Phase 1: Core Goal Execution
- **Data Models**: `Goal`, `Task`, `GoalSession`, `SuccessEvaluation` models
- **Goal Executor**: `GoalExecutor` class with basic task execution loop
- **Task States**: State machine for `pending`, `in_progress`, `completed`, `failed`
- **Codebot Integration**: `await codebot.execute()` with mode selection
- **CLI Commands**: `achieve`, `status`, `resume` with Click implementation

### Phase 2: Claude SDK Integration
- **Claude SDK Context**: `ClaudeSDKContext` class for claude -p executions
- **Task Execution**: Each task runs autonomously to completion
- **Result Extraction**: Parse file changes from claude -p output
- **Progress Tracking**: `iteration_count`, `completed_tasks` persistence

### Phase 3: Task Planning Intelligence
- **Task Planner Agent**: PydanticAI agent with `TaskPlan` result type
- **Task Dependencies**: `Task.dependencies` and `Task.is_ready()` logic
- **Dynamic Replanning**: `_generate_additional_tasks()` when blocked
- **Success Evaluator**: PydanticAI agent for criteria checking with confidence scores