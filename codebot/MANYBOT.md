# Manybot: Autonomous Agents with Goals & Responsibilities

Manybots are persistent AI agents that work autonomously toward measurable goals within defined areas of responsibility, inspired by tech company organizational principles like OKRs and DRIs.

## Architecture Overview

Manybot is a client/server system that orchestrates autonomous development agents:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Manybot CLI   │     │   Manybot GUI    │     │  GitHub/GitLab  │
│  (manybot cmd)  │     │  (Future Web UI)  │     │   (Webhooks)    │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Manybot Server        │
                    │  (Background Service)   │
                    │                         │
                    │  • Bot Registry         │
                    │  • Event Queue          │
                    │  • Work Scheduler       │
                    │  • Coordination Engine  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Codebot Library       │
                    │  (Core Operations)      │
                    │                         │
                    │  • Commands             │
                    │  • Context Management   │
                    │  • File Operations      │
                    │  • Git Integration      │
                    └─────────────────────────┘
```

The Manybot server uses the Codebot library to execute actual code operations, while adding:
- Persistent state management
- Multi-agent coordination
- Event-driven execution
- Background work loops
- Goal tracking and KRs

## Quick Start

```bash
# Create a test coverage bot with specific KRs
manybot create test-guardian \
  --goal "Q1: Achieve comprehensive test coverage" \
  --kr "Increase unit test coverage from 67% to 90%" \
  --kr "100% coverage for all public API endpoints \
  --kr "Add integration tests for all user workflows" \
  --owns "src/api/" "tests/" \
  --reviewer "human:tech-lead"

# Create a performance optimization bot
manybot create perf-optimizer \
  --goal "Reduce API latency by 50%" \
  --kr "p50 latency < 100ms for all endpoints" \
  --kr "p99 latency < 500ms under load" \
  --timeline "6 weeks" \
  --owns "src/api/" "src/db/" \
  --metric "pytest-benchmark"
```

## Core Concepts

### Goals: Measurable Objectives with Timelines

Goals follow the OKR (Objectives and Key Results) pattern:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum

class TimeHorizon(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class KeyResult(BaseModel):
    """Measurable outcome contributing to a goal"""
    description: str
    target_value: float
    current_value: float = 0.0
    unit: str  # "percent", "milliseconds", "count", etc.
    measurement_method: str  # Python expression or function
    deadline: datetime

    @property
    def progress(self) -> float:
        """Calculate progress as percentage"""
        if self.target_value == 0:
            return 100.0
        return (self.current_value / self.target_value) * 100

    @property
    def is_on_track(self) -> bool:
        """Check if KR is on track based on time elapsed"""
        time_elapsed = (datetime.now() - self.created_at).days
        total_time = (self.deadline - self.created_at).days
        expected_progress = (time_elapsed / total_time) * 100 if total_time > 0 else 0
        return self.progress >= expected_progress * 0.9  # 90% of expected

class Goal(BaseModel):
    """High-level objective with measurable key results"""
    id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    objective: str  # "Improve system reliability"
    key_results: List[KeyResult]
    rationale: str  # Why this matters
    created_by: str  # "human:alice" or "manybot:parent-bot"
    created_at: datetime = Field(default_factory=datetime.now)
    time_horizon: TimeHorizon = TimeHorizon.QUARTERLY

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all KRs and return current state"""
        return {
            "objective": self.objective,
            "overall_progress": sum(kr.progress for kr in self.key_results) / len(self.key_results),
            "key_results": [
                {
                    "description": kr.description,
                    "progress": kr.progress,
                    "on_track": kr.is_on_track,
                    "current": f"{kr.current_value} {kr.unit}",
                    "target": f"{kr.target_value} {kr.unit}"
                }
                for kr in self.key_results
            ],
            "at_risk": any(not kr.is_on_track for kr in self.key_results)
        }
```

### Responsibilities: Ownership & Authority

Responsibilities define what parts of the codebase a bot owns and can modify:

```python
class ResponsibilityLevel(str, Enum):
    OBSERVER = "observer"      # Read-only, can suggest changes
    CONTRIBUTOR = "contributor" # Can make changes with review
    MAINTAINER = "maintainer"  # Can make changes, approve others
    OWNER = "owner"           # Full authority, can delegate

class Responsibility(BaseModel):
    """Defines ownership over code areas"""
    paths: List[str]  # Glob patterns like "src/api/**/*.py"
    level: ResponsibilityLevel
    description: str  # What this responsibility covers
    owner: str  # "manybot:bot-id" or "human:username"
    delegates: List[str] = Field(default_factory=list)

    # Review requirements
    review_required: bool = True
    auto_approve_patterns: List[str] = Field(default_factory=list)
    reviewers: List[str] = Field(default_factory=list)

    # Constraints
    max_changes_per_cycle: int = 10
    require_tests: bool = True
    coverage_threshold: Optional[float] = None

    def can_modify(self, filepath: str, author: str) -> bool:
        """Check if author can modify file"""
        if not any(Path(filepath).match(pattern) for pattern in self.paths):
            return False

        if author == self.owner:
            return True

        if author in self.delegates and self.level >= ResponsibilityLevel.CONTRIBUTOR:
            return True

        return False

    def needs_review(self, filepath: str) -> bool:
        """Check if changes need review"""
        if not self.review_required:
            return False

        for pattern in self.auto_approve_patterns:
            if Path(filepath).match(pattern):
                return False

        return True
```

### The do_work Loop: Autonomous Execution

The core execution loop that makes manybots autonomous:

```python
from pydantic_ai import Agent, ModelRetry
from pydantic import BaseModel
import asyncio

class WorkCycle(BaseModel):
    """Single iteration of autonomous work"""
    cycle_id: str
    bot_id: str
    timestamp: datetime

    # Analysis phase
    goal_evaluation: Dict[str, Any]
    responsibility_review: List[Dict[str, Any]]
    context_summary: str

    # Planning phase
    planned_tasks: List[str]
    priority_rationale: str

    # Execution phase
    executed_commands: List[str]
    changes_made: List[FileChange]
    test_results: Optional[Dict[str, Any]]

    # Reflection phase
    outcomes: Dict[str, Any]
    lessons_learned: List[str]
    next_cycle_focus: str

class Manybot:
    """Autonomous agent with goals and responsibilities
    
    Uses the Codebot library for all code operations while adding
    persistent state, goal tracking, and autonomous execution.
    """

    def __init__(self,
                 name: str,
                 goal: Goal,
                 responsibilities: List[Responsibility]):
        self.name = name
        self.goal = goal
        self.responsibilities = responsibilities
        self.state = ManybotState(
            bot_id=f"{name}_{uuid.uuid4().hex[:8]}",
            status="active"
        )
        
        # Import and use Codebot as a library
        from codebot import Command, ContextManager, CommandExecutor
        from codebot.workflows import WorkflowExecutor
        
        self.codebot_executor = CommandExecutor()
        self.context_manager = ContextManager()
        self.workflow_executor = WorkflowExecutor()

        # Initialize PydanticAI agent for planning
        self.planner = Agent(
            result_type=List[str],  # List of tasks
            system_prompt=self._build_planner_prompt()
        )

        # Initialize PydanticAI agent for reflection
        self.reflector = Agent(
            result_type=WorkCycleReflection,
            system_prompt="Analyze work outcomes and extract learnings"
        )

    async def do_work(self) -> WorkCycle:
        """Execute one autonomous work cycle"""
        cycle = WorkCycle(
            cycle_id=str(uuid.uuid4()),
            bot_id=self.state.bot_id,
            timestamp=datetime.now()
        )

        # 1. Analyze current state
        cycle.goal_evaluation = self.goal.evaluate()
        cycle.responsibility_review = await self._review_responsibilities()
        cycle.context_summary = await self._analyze_codebase()

        # 2. Plan tasks based on goals and responsibilities
        planning_context = {
            "goal_evaluation": cycle.goal_evaluation,
            "responsibility_review": cycle.responsibility_review,
            "recent_changes": self._get_recent_changes(),
            "blocked_tasks": self.state.blocked_tasks
        }

        result = await self.planner.run(planning_context)
        cycle.planned_tasks = result.data
        cycle.priority_rationale = result.new_messages[-1].content

        # 3. Execute tasks
        for task in cycle.planned_tasks[:3]:  # Limit tasks per cycle
            try:
                command = self._task_to_command(task)
                output = await command.execute(self._build_context())

                # Check if we can modify the files
                authorized_changes = []
                for change in output.file_changes:
                    if self._can_modify(change.filepath):
                        authorized_changes.append(change)
                    else:
                        # Create PR instead
                        await self._create_pr_for_change(change)

                # Apply authorized changes
                for change in authorized_changes:
                    await self._apply_change(change)
                    cycle.changes_made.append(change)

                cycle.executed_commands.append(command.name)

            except Exception as e:
                self.state.blocked_tasks.append({
                    "task": task,
                    "error": str(e),
                    "timestamp": datetime.now()
                })

        # 4. Run tests if changes were made
        if cycle.changes_made:
            cycle.test_results = await self._run_relevant_tests(cycle.changes_made)

        # 5. Reflect on outcomes
        reflection_input = WorkCycleReflectionInput(
            cycle=cycle,
            goal=self.goal,
            previous_cycles=self.state.work_history[-5:]
        )

        reflection = await self.reflector.run(reflection_input)
        cycle.outcomes = reflection.data.outcomes
        cycle.lessons_learned = reflection.data.lessons
        cycle.next_cycle_focus = reflection.data.next_focus

        # 6. Update state
        self.state.work_history.append(cycle)
        self._update_goal_progress(cycle)

        return cycle

    def _build_planner_prompt(self) -> str:
        return f"""You are {self.name}, an autonomous development agent.

Your goal: {self.goal.objective}

Your areas of responsibility:
{chr(10).join(f'- {resp.description}: {", ".join(resp.paths)}' for resp in self.responsibilities)}

When planning tasks:
1. Focus on making measurable progress toward key results
2. Respect ownership boundaries - only modify files you own
3. Prioritize tasks that are failing or at risk
4. Balance quick wins with substantial progress
5. Consider dependencies and blockers

Output a prioritized list of specific, actionable tasks."""
```

### Two Execution Models: Background Loop vs Event-Triggered

Manybots support two complementary execution models:

```python
class ManybotExecutionMode(str, Enum):
    CONTINUOUS = "continuous"  # Background loop
    TRIGGERED = "triggered"    # Event-driven

class Manybot:
    """Supports both continuous background execution and event-triggered workflows"""
    
    def __init__(self, name: str, goal: Goal, responsibilities: List[Responsibility],
                 mode: ManybotExecutionMode = ManybotExecutionMode.CONTINUOUS):
        self.mode = mode
        self.event_queue: List[GitHubEvent] = []
        # ... rest of init
    
    async def run(self):
        """Main entry point - routes to appropriate execution model"""
        if self.mode == ManybotExecutionMode.CONTINUOUS:
            await self.run_continuous_loop()
        else:
            await self.run_event_driven()
    
    async def run_continuous_loop(self):
        """Background loop API - autonomous continuous execution"""
        while self.state.status == "active":
            # Check for any queued events first
            if self.event_queue:
                await self._process_events()
            
            # Run regular work cycle
            cycle = await self.do_work()
            
            # Adaptive sleep
            sleep_time = self._calculate_sleep_time(cycle)
            await asyncio.sleep(sleep_time)
    
    async def run_event_driven(self):
        """PubSub triggered API - only runs when events occur"""
        while self.state.status == "active":
            # Wait for events
            event = await self._wait_for_event()
            
            # Process event immediately
            await self._handle_event(event)
            
            # Run work cycle in response
            await self.do_work()
    
    async def _handle_event(self, event: GitHubEvent):
        """Process GitHub events and update task queue"""
        if isinstance(event, PushEvent):
            # Add review task for changes
            self.state.current_tasks.insert(0, 
                f"Review and test changes in {event.branch}")
        elif isinstance(event, PREvent):
            # Add PR review task
            self.state.current_tasks.insert(0,
                f"Review PR #{event.pr_number}: {event.title}")
        elif isinstance(event, IssueCommentEvent):
            # Parse commands from comments
            if "@manybot" in event.comment:
                command = self._parse_command(event.comment)
                self.state.current_tasks.insert(0, command)

# Usage examples:

# Continuous background bot
test_bot = Manybot(
    name="test-guardian",
    goal=test_coverage_goal,
    responsibilities=test_responsibilities,
    mode=ManybotExecutionMode.CONTINUOUS
)
asyncio.create_task(test_bot.run())  # Runs forever

# Event-triggered bot
pr_review_bot = Manybot(
    name="pr-reviewer", 
    goal=code_quality_goal,
    responsibilities=review_responsibilities,
    mode=ManybotExecutionMode.TRIGGERED
)
asyncio.create_task(pr_review_bot.run())  # Waits for events

# Hybrid bot (continuous with event priority)
hybrid_bot = Manybot(
    name="hybrid-worker",
    goal=maintenance_goal,
    responsibilities=all_responsibilities,
    mode=ManybotExecutionMode.CONTINUOUS
)
# This bot runs continuously but prioritizes events when they arrive
```

### Bot Coordination: Working Together

Manybots can coordinate through structured meetings:

```python
class MeetingAgenda(BaseModel):
    """Structured agenda for bot coordination"""
    meeting_id: str
    organizer: str  # Bot that called the meeting
    attendees: List[str]  # Bot IDs
    topic: str
    context: Dict[str, Any]
    decisions_needed: List[str]

class MeetingOutcome(BaseModel):
    """Results from bot coordination meeting"""
    decisions: Dict[str, str]
    action_items: List[ActionItem]
    follow_up_meetings: List[MeetingAgenda]
    responsibility_changes: List[ResponsibilityChange]

class BotCoordinator:
    """Facilitates coordination between multiple bots"""

    async def hold_meeting(self, agenda: MeetingAgenda) -> MeetingOutcome:
        """Simulate a coordination meeting between bots"""

        # Each bot analyzes the agenda from their perspective
        bot_inputs = []
        for bot_id in agenda.attendees:
            bot = self.registry.get_bot(bot_id)
            analysis = await bot.analyze_agenda(agenda)
            bot_inputs.append({
                "bot_id": bot_id,
                "goal": bot.goal.objective,
                "relevant_responsibilities": [resp.description for resp in bot.responsibilities],
                "perspective": analysis
            })

        # Use PydanticAI agent to facilitate discussion
        facilitator = Agent(
            result_type=MeetingOutcome,
            system_prompt="""You are facilitating a meeting between autonomous bots.
            Help them coordinate their efforts, resolve conflicts, and make decisions
            that advance their collective goals while respecting individual responsibilities."""
        )

        meeting_context = {
            "agenda": agenda,
            "bot_inputs": bot_inputs,
            "shared_codebase_state": await self._get_codebase_state()
        }

        result = await facilitator.run(meeting_context)
        outcome = result.data

        # Apply decisions
        await self._apply_meeting_outcomes(outcome)

        return outcome
```

### GitHub PubSub Integration

Filesystem changes trigger automated workflows through GitHub events:

```python
class GitHubPubSubHandler:
    """Connects filesystem changes to manybot workflows"""
    
    def __init__(self):
        self.path_to_bot = self._build_ownership_map()
    
    async def on_file_change(self, event: GitHubFileChangeEvent):
        """Route file changes to responsible manybots"""
        affected_path = event.file_path
        
        # Find responsible manybot
        responsible_bot = self._find_responsible_bot(affected_path)
        
        if responsible_bot:
            # Trigger appropriate workflow based on change type
            if event.is_in_owned_path:
                await self._trigger_review_workflow(responsible_bot, event)
            elif event.is_dependency:
                await self._trigger_iterate_workflow(responsible_bot, event)
            elif event.affects_tests:
                await self._trigger_test_workflow(responsible_bot, event)
    
    async def _trigger_review_workflow(self, bot_id: str, event: GitHubFileChangeEvent):
        """Someone changed files in bot's responsibility - trigger review"""
        bot = await self.load_manybot(bot_id)
        
        # Special review task in next work cycle
        review_task = f"Review external changes to {event.file_path}"
        bot.state.current_tasks.insert(0, review_task)
        
        # Optionally trigger immediate work cycle
        if event.is_critical:
            await bot.do_work()
```

### Remote Execution & Subscriptions

Manybots can subscribe to repository events:

```python
class Subscription(BaseModel):
    """Subscribe to repository events"""
    id: str
    repo: str
    branch_pattern: str = "main"
    paths: List[str]
    bot_id: str
    trigger: Literal["push", "pr", "schedule", "issue_comment"]
    config: Dict[str, Any] = {}

class RemoteExecutor:
    """Handles remote manybot execution"""
    
    async def handle_github_webhook(self, event: GitHubEvent):
        # Find matching subscriptions
        subscriptions = await self.find_matching_subscriptions(
            repo=event.repository,
            paths=event.changed_files,
            trigger=event.event_type
        )
        
        # Queue bot work cycles
        for sub in subscriptions:
            bot = await self.get_bot(sub.bot_id)
            # Add event to bot's context
            bot.state.github_events.append(event)
            # Trigger work cycle
            await bot.do_work()
```

### OWNERS File Integration

Filesystem-based responsibility representation:

```yaml
# src/api/OWNERS
primary: manybot:api-bot
delegates:
  - manybot:test-bot    # Can modify test files
  - human:alice         # Senior engineer

reviewers:
  required:
    - human:tech-lead   # For breaking changes
  optional:
    - manybot:security-bot
    - manybot:perf-bot

auto_approve:
  - "*.test.py"
  - "test_*.py"
  - "*/fixtures/*"

policies:
  max_changes_per_pr: 20
  require_tests: true
  min_coverage: 85

# Meetings this area participates in
meetings:
  - api_design_review   # Weekly
  - system_architecture # Monthly
```

## Implementation with PydanticAI

### Creating a Manybot

```python
from pydantic_ai import Agent
from typing import AsyncIterator
import asyncio

async def create_manybot(
    name: str,
    goal_description: str,
    paths: List[str],
    parent: Optional[str] = None
) -> Manybot:
    """Create a new manybot with goal and responsibility planning"""

    # Use AI to help structure the goal
    goal_planner = Agent(
        result_type=Goal,
        system_prompt="""Help create a well-structured goal with measurable KRs.
        Goals should be specific, measurable, achievable, relevant, and time-bound.
        Break down high-level objectives into 2-4 concrete key results."""
    )

    goal_context = {
        "description": goal_description,
        "codebase_analysis": await analyze_paths(paths),
        "parent_goal": parent.goal if parent else None
    }

    result = await goal_planner.run(goal_context)
    goal = result.data

    # Define responsibilities based on paths
    responsibilities = []
    for path in paths:
        responsibility = Responsibility(
            paths=[path],
            level=ResponsibilityLevel.MAINTAINER,
            description=f"Maintain and improve {path}",
            owner=f"manybot:{name}",
            review_required=True,
            auto_approve_patterns=["*.test.py", "test_*.py"]
        )
        responsibilities.append(responsibility)

    # Create the bot
    bot = Manybot(name=name, goal=goal, responsibilities=responsibilities)

    # Register in global registry
    BotRegistry.instance().register(bot)

    # Start autonomous execution
    asyncio.create_task(run_bot_loop(bot))

    return bot

async def run_bot_loop(bot: Manybot):
    """Run bot's background loop continuously
    
    This is the core innovation - transforming triggered workflows into
    continuous autonomous loops that work toward goals.
    """
    # Use the continuous loop API
    await bot.run_continuous_loop()
```

### Bot Evolution: Meetings & Handoffs

```python
class ManybotEvolution:
    """Handles bot lifecycle, meetings, and responsibility transfers"""

    async def quarterly_review(self) -> None:
        """Review all bots' progress and adjust goals/responsibilities"""

        active_bots = BotRegistry.instance().get_active_bots()

        # Analyze collective progress
        review_agent = Agent(
            result_type=QuarterlyReview,
            system_prompt="Analyze bot ecosystem health and recommend adjustments"
        )

        review = await review_agent.run({
            "bots": [bot.to_dict() for bot in active_bots],
            "codebase_metrics": await get_codebase_metrics(),
            "incident_reports": await get_incident_reports()
        })

        # Hold coordination meeting
        if review.data.coordination_needed:
            agenda = MeetingAgenda(
                meeting_id=f"quarterly_review_{datetime.now().strftime('%Y_Q%Q')}",
                organizer="system",
                attendees=[bot.bot_id for bot in active_bots],
                topic="Quarterly goal and responsibility alignment",
                context=review.data.context,
                decisions_needed=review.data.decisions_needed
            )

            outcome = await BotCoordinator().hold_meeting(agenda)

            # Apply changes
            for change in outcome.responsibility_changes:
                await self._apply_responsibility_change(change)

    async def spawn_specialist(self,
                             parent_bot: Manybot,
                             specialty: str) -> Manybot:
        """Parent bot creates a specialist child bot"""

        # Parent bot decides on child's goal
        child_goal_agent = Agent(
            result_type=Goal,
            system_prompt=f"Create a focused sub-goal for a specialist bot"
        )

        child_goal = await child_goal_agent.run({
            "parent_goal": parent_bot.goal,
            "specialty": specialty,
            "parent_progress": parent_bot.goal.evaluate(),
            "available_paths": await self._find_unowned_paths()
        })

        # Create child with delegated responsibility
        child_bot = await create_manybot(
            name=f"{parent_bot.name}_{specialty}",
            goal_description=child_goal.data.objective,
            paths=child_goal.data.suggested_paths,
            parent=parent_bot
        )

        # Update OWNERS files
        for path in child_goal.data.suggested_paths:
            await update_owners_file(
                path=path,
                owner=f"manybot:{child_bot.bot_id}",
                delegates=[f"manybot:{parent_bot.bot_id}"]
            )

        return child_bot
```

## Real-World Example

```python
# Example: Creating a comprehensive test infrastructure bot
async def setup_test_infrastructure_bot():
    bot = await create_manybot(
        name="test-infra-bot",
        goal_description="""Q1 2024: Establish world-class test infrastructure
        - Increase overall test coverage from 67% to 90%
        - Reduce test flakiness to < 1%
        - Improve test execution time by 50%
        - Ensure all PRs have test coverage""",
        paths=["tests/", "src/", ".github/workflows/", "pytest.ini"]
    )

    # Bot's first work cycle might:
    # 1. Analyze current test coverage gaps
    # 2. Generate missing unit tests for uncovered code
    # 3. Add integration tests for critical paths
    # 4. Set up coverage reporting in CI
    # 5. Create dashboard for tracking progress

    # After a few cycles, it might spawn specialists:
    perf_test_bot = await bot.spawn_specialist("performance_testing")
    flaky_test_bot = await bot.spawn_specialist("flaky_test_detection")

    return bot
```

## CLI Integration

### Managing Subscriptions

```bash
# Subscribe to events
manybot subscribe test-bot src/ tests/          # Test on any change to src/ or tests/
manybot subscribe lint-bot --paths "*.py"       # Lint all Python files
manybot subscribe review-bot --trigger pr       # Review on PR creation/update

# Manage subscriptions
manybot subscriptions list                       # List all subscriptions
manybot subscriptions list --repo myorg/myrepo   # List for specific repo
manybot subscriptions show <subscription-id>     # Show subscription details
manybot subscriptions pause <subscription-id>    # Pause subscription
manybot subscriptions resume <subscription-id>   # Resume subscription
manybot subscriptions delete <subscription-id>   # Delete subscription

# View execution history
manybot logs <bot-id>                           # View recent work cycles
manybot logs <bot-id> --tail 50                 # View last 50 log entries
manybot result <cycle-id>                        # View specific cycle result

# Manual trigger
manybot trigger <bot-id>                         # Manually trigger work cycle
```

## Configuration

```python
# ~/.manybot/config.py
from pydantic import BaseSettings

class ManybotConfig(BaseSettings):
    # Work cycle settings
    max_tasks_per_cycle: int = 5
    min_sleep_seconds: int = 300
    max_sleep_seconds: int = 3600

    # Goal settings
    default_time_horizon: str = "quarterly"
    kr_warning_threshold: float = 0.8  # Warn if < 80% progress expected

    # Responsibility settings
    require_owners_files: bool = True
    max_changes_without_review: int = 10
    auto_approve_test_changes: bool = True

    # Coordination settings
    enable_bot_meetings: bool = True
    meeting_frequency_days: int = 7
    require_human_oversight: bool = True

    # Safety settings
    dry_run_mode: bool = False
    rollback_on_test_failure: bool = True
    max_bots_per_goal: int = 10

    class Config:
        env_prefix = "MANYBOT_"
```

## Server Architecture

### Manybot Server Components

```python
from fastapi import FastAPI, WebSocket
from typing import Dict, List
import asyncio

class ManybotServer:
    """Central server managing all manybots"""
    
    def __init__(self):
        self.app = FastAPI()
        self.bots: Dict[str, Manybot] = {}
        self.event_queue = asyncio.Queue()
        self.websocket_clients: List[WebSocket] = []
        
        # Setup API routes
        self._setup_routes()
        
    def _setup_routes(self):
        """API endpoints for CLI and GUI clients"""
        
        @self.app.post("/bots")
        async def create_bot(bot_config: BotConfig):
            """Create new manybot"""
            bot = await self._create_bot(bot_config)
            asyncio.create_task(bot.run())
            return {"bot_id": bot.state.bot_id}
        
        @self.app.get("/bots")
        async def list_bots():
            """List all active bots"""
            return [bot.to_summary() for bot in self.bots.values()]
        
        @self.app.get("/bots/{bot_id}/status")
        async def bot_status(bot_id: str):
            """Get detailed bot status"""
            bot = self.bots.get(bot_id)
            return bot.get_status() if bot else None
        
        @self.app.post("/bots/{bot_id}/trigger")
        async def trigger_bot(bot_id: str):
            """Manually trigger bot work cycle"""
            bot = self.bots.get(bot_id)
            if bot:
                cycle = await bot.do_work()
                return cycle.model_dump()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Real-time updates for GUI"""
            await websocket.accept()
            self.websocket_clients.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            finally:
                self.websocket_clients.remove(websocket)
    
    async def _create_bot(self, config: BotConfig) -> Manybot:
        """Create bot using codebot library"""
        # Use codebot to validate paths exist
        from codebot import ContextManager
        context = ContextManager()
        
        # Verify paths are valid
        for path in config.paths:
            if not context.validate_path(path):
                raise ValueError(f"Invalid path: {path}")
        
        # Create manybot instance
        bot = Manybot(
            name=config.name,
            goal=config.goal,
            responsibilities=self._create_responsibilities(config)
        )
        
        self.bots[bot.state.bot_id] = bot
        return bot
    
    async def handle_github_webhook(self, event: GitHubEvent):
        """Process GitHub webhooks"""
        await self.event_queue.put(event)
        
        # Route to appropriate bots
        for bot in self.bots.values():
            if self._event_matches_bot(event, bot):
                bot.event_queue.append(event)
    
    async def broadcast_update(self, update: Dict):
        """Send updates to all connected clients"""
        for client in self.websocket_clients:
            await client.send_json(update)
```

### CLI Client

```python
import click
import httpx
import asyncio

class ManybotCLI:
    """CLI client for manybot server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client = httpx.AsyncClient()
    
    @click.group()
    def cli():
        """Manybot management CLI"""
        pass
    
    @cli.command()
    @click.option('--name', required=True)
    @click.option('--goal', required=True)
    @click.option('--owns', multiple=True, required=True)
    async def create(name: str, goal: str, owns: tuple):
        """Create a new manybot"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/bots",
                json={
                    "name": name,
                    "goal_description": goal,
                    "paths": list(owns)
                }
            )
            bot_data = response.json()
            print(f"Created bot: {bot_data['bot_id']}")
    
    @cli.command()
    async def list():
        """List all active bots"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.server_url}/bots")
            bots = response.json()
            for bot in bots:
                print(f"{bot['name']} - {bot['status']}")
                print(f"  Goal: {bot['goal']['objective']}")
                print(f"  Progress: {bot['goal']['overall_progress']}%")
```

## Implementation Roadmap

1. **Phase 1**: Basic do_work loop with simple goal tracking
   - Implement core Manybot class using codebot library
   - Single bot execution with CLI

2. **Phase 2**: Client/Server Architecture
   - FastAPI server with bot registry
   - RESTful API for bot management
   - Persistent state storage

3. **Phase 3**: Multi-bot coordination
   - Event queue and routing
   - Bot meetings and coordination
   - OWNERS file integration

4. **Phase 4**: Advanced Features
   - WebSocket support for real-time updates
   - Web GUI for monitoring
   - Bot spawning and delegation
   - GitHub/GitLab webhook integration

The system builds incrementally from simple autonomous loops to a full client/server architecture for coordinated multi-agent development.
