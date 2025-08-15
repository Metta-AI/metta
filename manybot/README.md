# Manybot

```bash
# Create bots with specific goals
manybot create api-modernizer --goal "Modernize API to async/await" --success "All endpoints use async"

# Assign responsibilities
manybot assign test-guardian --owns "src/api/" "tests/api/" --level owner

# Schedule coordination
manybot meeting schedule "API redesign" --attendees api-bot,test-bot,docs-bot

# Query status
manybot status test-guardian
```

## Coordinated Multi-Agent System

Manybot orchestrates teams of self-directed agents working toward shared objectives:

- **Bot Management**: Create and track bots with specific goals
- **Responsibility Assignment**: Map code ownership through OWNERS files
- **Bot Communication**: Structured meetings and decision-making protocols
- **Progress Tracking**: Monitor goal achievement and bot activities
- **Self-Direction**: Bots can spawn specialists, update goals, and delegate (with PR review)

## Diamond Architecture

```
                    ┌─────────────┐
                    │   manybot   │
                    │             │
                    │ • Bot teams │
                    │ • Meetings  │
                    │ • OWNERS    │
                    └──────┬──────┘
                          /\
                         /  \
                        /    \
               ┌───────▼─┐  ┌─▼────────┐
               │ goalbot │  │remotebot │
               │         │  │          │
               │ • Goals │  │ • Servers│
               │ • Tasks │  │ • Jobs   │
               │ • Loop  │  │ • Deploy │
               └────┬────┘  └────┬─────┘
                    \            /
                     \          /
                      \        /
                    ┌──▼──────▼──┐
                    │   codebot   │
                    │             │
                    │ • FileChange│
                    │ • Commands  │
                    │ • Context   │
                    └─────────────┘
```

**Data flows up, services flow down:**
- **codebot**: Foundation - all file operations happen here
- **goalbot**: Adds persistence and goal-driven loops on top of codebot
- **remotebot**: Provides distributed execution infrastructure
- **manybot**: Orchestrates multiple goal-driven bots across infrastructure

## Service Interface

### Provides
- Bot identity management and registry
- Responsibility assignment and OWNERS files
- Inter-bot communication protocols
- Progress monitoring and reporting

### Consumes
- goalbot: For autonomous goal execution
- remotebot: For distributed bot deployment

## Core Operations

### Creating and Managing Bots

```python
class CreateBot(BaseModel):
    """Create a new bot with specific goal"""
    name: str
    goal: str
    success_criteria: str
    
    async def execute(self) -> BotHandle:
        # Register bot in manybot's registry
        bot = Bot(
            id=f"bot_{self.name}",
            goal=Goal(
                description=self.goal,
                success_criteria=self.success_criteria
            ),
            status="active"
        )
        
        await BotRegistry.register(bot)
        
        # Bot will start working via goalbot
        await self.start_work_cycle(bot)
        
        return BotHandle(bot_id=bot.id)

# Usage
await manybot.create_bot(
    name="api-modernizer",
    goal="Modernize API to use async/await patterns",
    success_criteria="All API endpoints use async handlers"
)
```

### Managing Responsibilities

```python
class AssignResponsibility(BaseModel):
    """Assign code ownership to a bot"""
    bot_id: str
    paths: List[str]  # Glob patterns
    level: Literal["owner", "maintainer", "contributor"]
    
    async def execute(self) -> None:
        # Update OWNERS file
        for path in self.paths:
            owners_file = Path(path) / "OWNERS"
            ownership = Ownership(
                primary=f"manybot:{self.bot_id}",
                level=self.level
            )
            await self.update_owners_file(owners_file, ownership)
        
        # Notify bot of new responsibility
        bot = await BotRegistry.get(self.bot_id)
        await bot.add_responsibility(
            Responsibility(
                paths=self.paths,
                level=self.level,
                owner=f"manybot:{self.bot_id}"
            )
        )

# Usage
await manybot.assign_responsibility(
    bot_id="test-guardian",
    paths=["src/api/**/*.py", "tests/api/**/*.py"],
    level="owner"
)
```

### Bot Coordination Models

```python
class MeetingAgenda(BaseModel):
    """Structured agenda for bot coordination"""
    meeting_id: str
    organizer: str  # Bot that called the meeting
    attendees: List[str]  # Bot IDs
    topic: str
    context: Dict[str, Any]
    decisions_needed: List[str]
    time_limit_minutes: int = 30

class MeetingOutcome(BaseModel):
    """Results from bot coordination meeting"""
    decisions: Dict[str, str]
    action_items: List[ActionItem]
    follow_up_meetings: List[MeetingAgenda]
    responsibility_changes: List[ResponsibilityChange]

class ActionItem(BaseModel):
    """Task assigned during meeting"""
    assigned_to: str  # Bot ID
    task: str
    deadline: datetime
    priority: Literal["high", "medium", "low"]

class ResponsibilityChange(BaseModel):
    """Change in code ownership"""
    path: str
    from_owner: str
    to_owner: str
    level: Literal["owner", "maintainer", "contributor"]
    reason: str
```

## Bot Coordination Implementation

```python
class BotCoordinator:
    """Facilitates coordination between multiple bots"""
    
    async def hold_meeting(self, agenda: MeetingAgenda) -> MeetingOutcome:
        """Simulate a coordination meeting between bots using PydanticAI"""
        
        # Gather bot perspectives
        bot_inputs = []
        for bot_id in agenda.attendees:
            bot = await BotRegistry.get(bot_id)
            analysis = await bot.analyze_agenda(agenda)
            bot_inputs.append({
                "bot_id": bot_id,
                "goal": bot.goal.objective,
                "owned_paths": bot.get_owned_paths(),
                "perspective": analysis
            })
        
        # Use PydanticAI agent to facilitate
        facilitator = Agent(
            result_type=MeetingOutcome,
            system_prompt="""You are facilitating a meeting between autonomous bots.
            Help them coordinate efforts, resolve conflicts, and make decisions
            that advance their collective goals while respecting individual responsibilities."""
        )
        
        meeting_context = {
            "agenda": agenda.model_dump(),
            "bot_inputs": bot_inputs,
            "codebase_state": await self._get_codebase_state()
        }
        
        result = await facilitator.run(meeting_context)
        outcome = result.data
        
        # Apply decisions
        await self._apply_meeting_outcomes(outcome)
        
        return outcome

class ScheduleMeeting(BaseModel):
    """Schedule coordination between bots"""
    topic: str
    attendees: List[str]  # Bot IDs
    decisions_needed: List[str]
    context: Dict[str, Any] = Field(default_factory=dict)
    
    async def execute(self) -> MeetingHandle:
        # Create structured agenda
        agenda = MeetingAgenda(
            meeting_id=f"meeting_{uuid4()}",
            organizer="system",  # Or requesting bot
            attendees=self.attendees,
            topic=self.topic,
            context=self.context,
            decisions_needed=self.decisions_needed
        )
        
        # Schedule meeting
        coordinator = BotCoordinator()
        outcome = await coordinator.hold_meeting(agenda)
        
        # Create handle for tracking
        return MeetingHandle(
            meeting_id=agenda.meeting_id,
            outcome=outcome
        )

# Usage  
meeting = await manybot.schedule_meeting(
    topic="Coordinate API redesign", 
    attendees=["api-bot", "test-bot", "docs-bot"],
    decisions_needed=[
        "API versioning strategy",
        "Breaking change timeline",
        "Documentation update plan"
    ],
    context={"proposed_changes": [...], "current_api_version": "v2"}
)
```

### Querying Bot Status

```python
class BotStatus(BaseModel):
    """Get current status of a bot"""
    bot_id: str
    
    async def execute(self) -> BotReport:
        bot = await BotRegistry.get(self.bot_id)
        
        # Get current work from goalbot
        current_goal = await bot.get_current_goal()
        progress = await current_goal.evaluate_progress()
        
        # Get owned files
        owned_paths = await self.get_owned_paths(bot.bot_id)
        
        return BotReport(
            bot_id=self.bot_id,
            status=bot.status,
            current_goal=current_goal.description,
            progress=progress,
            owned_paths=owned_paths,
            recent_changes=await bot.get_recent_changes()
        )

# Usage
status = await manybot.bot_status("test-guardian")
print(f"Progress on goal: {status.progress}%")
print(f"Owns: {', '.join(status.owned_paths)}")
```

## Data Flow Example: Fix Failing Tests

```python
# 1. Manybot receives GitHub webhook
webhook = GitHubPushEvent(
    branch="main",
    failed_tests=["test_api.py::test_auth"]
)

# 2. Manybot creates goal for responsible bot
bot = await manybot.find_owner("tests/test_api.py")
goal = Goal(
    description="Fix failing auth test",
    success_criteria="test_api.py::test_auth passes",
    context_paths=["src/api/auth.py", "tests/test_api.py"]
)

# 3. Bot executes goal via goalbot (may use remotebot)
session = await goalbot.execute(
    goal=goal,
    bot_id=bot.id,
    mode="sdk"  # Autonomous execution
)

# 4. Goalbot uses codebot for file operations
async for task in session.tasks():
    if task.type == "fix_code":
        result = await codebot.execute(
            command="debug-tests",
            context=ExecutionContext(
                clipboard=task.test_output,
                files=task.relevant_files
            ),
            mode="claudesdk"  # Autonomous execution
        )
        
        # Apply FileChanges
        for change in result.file_changes:
            change.apply()
```

## GitHub Integration

```python
class GitHubCoordinator:
    """Maps GitHub events to bot coordination"""

    def __init__(self):
        self.ownership_map = self._build_ownership_map()
        self.event_handlers = self._setup_handlers()

    async def handle_push_event(self, event: PushEvent):
        """Coordinate response to code push"""

        # Find affected bots
        affected_bots = self._find_affected_bots(event.changed_files)

        # Notify owner bots
        for bot_id, files in affected_bots.items():
            await self._notify_bot(
                bot_id,
                EventNotification(
                    type="external_change",
                    files=files,
                    author=event.author,
                    priority="high" if bot_id in self._get_owners(files) else "medium"
                )
            )

        # Schedule coordination if multiple bots affected
        if len(affected_bots) > 1:
            await self._schedule_coordination_meeting(
                topic=f"Coordinate response to changes in {event.branch}",
                attendees=list(affected_bots.keys()),
                context={"event": event}
            )

    async def handle_pr_event(self, event: PREvent):
        """Coordinate PR review"""

        # Find reviewers based on OWNERS
        reviewers = await self._find_reviewers(event.changed_files)

        # Create review tasks
        for reviewer in reviewers:
            await self._create_review_task(
                reviewer,
                PRReviewTask(
                    pr_number=event.pr_number,
                    priority=self._calculate_review_priority(reviewer, event)
                )
            )
```

## OWNERS File System

```yaml
# src/api/OWNERS
primary: manybot:api-bot
delegates:
  - manybot:test-bot
  - human:alice

auto_approve:
  - "*.test.py"
  - "test_*.py"
```

## Bot Lifecycle & State Management

```python
class BotState(BaseModel):
    """Persistent state for a manybot"""
    bot_id: str
    name: str
    status: Literal["active", "paused", "terminated"]
    created_at: datetime
    last_active: datetime
    
    # Work tracking
    goals_completed: int = 0
    current_goal: Optional[Goal] = None
    work_cycles: List[WorkCycleRecord] = Field(default_factory=list)
    
    # Relationships
    parent_bot: Optional[str] = None
    child_bots: List[str] = Field(default_factory=list)
    delegated_from: Dict[str, List[str]] = Field(default_factory=dict)  # bot_id -> paths

class WorkCycleRecord(BaseModel):
    """Record of a single work cycle"""
    cycle_id: str
    timestamp: datetime
    goal_progress: float
    files_changed: List[str]
    commands_executed: List[str]
    outcome: Literal["success", "partial", "blocked"]
    blockers: List[str] = Field(default_factory=list)
```

## Self-Direction & Evolution

```python
class BotEvolution:
    """Bots evolve their capabilities and organization"""

    async def spawn_specialist(self, parent_bot: Manybot, need: str) -> Manybot:
        """Parent bot creates specialist for identified need"""
        
        # Analyze need and define specialist goal
        specialist_goal = await self._define_specialist_goal(
            parent_goal=parent_bot.goal,
            specialty_need=need,
            available_paths=await self._find_unowned_paths()
        )
        
        # Create specialist
        specialist = await manybot.create_bot(
            name=f"{parent_bot.name}_{need}",
            goal=specialist_goal.description,
            success_criteria=specialist_goal.success_criteria
        )
        
        # Establish relationship
        parent_bot.state.child_bots.append(specialist.bot_id)
        specialist.state.parent_bot = parent_bot.bot_id
        
        # Delegate specific paths
        delegated_paths = await self._determine_delegation(parent_bot, need)
        for path in delegated_paths:
            await manybot.delegate_ownership(
                from_bot=parent_bot.bot_id,
                to_bot=specialist.bot_id,
                paths=[path],
                retain_oversight=True
            )
        
        return specialist
    
    async def merge_bots(self, bot_ids: List[str], new_name: str) -> Manybot:
        """Merge multiple bots into one with combined responsibilities"""
        bots = [await BotRegistry.get(bot_id) for bot_id in bot_ids]
        
        # Combine goals and responsibilities
        merged_goal = await self._synthesize_goal(bots)
        merged_paths = []
        for bot in bots:
            merged_paths.extend(bot.get_owned_paths())
        
        # Create merged bot
        merged = await manybot.create_bot(
            name=new_name,
            goal=merged_goal.description,
            success_criteria=merged_goal.success_criteria
        )
        
        # Transfer responsibilities
        for path in merged_paths:
            await manybot.transfer_ownership(
                from_bot=bot.bot_id,
                to_bot=merged.bot_id,
                path=path
            )
        
        # Archive original bots
        for bot in bots:
            bot.state.status = "terminated"
            bot.state.metadata["merged_into"] = merged.bot_id
        
        return merged
```

## Bot Evolution & Delegation

```python
class SpawnSpecialist(BaseModel):
    """Parent bot creates specialist for specific need"""
    parent_bot_id: str
    specialty: str
    delegated_paths: List[str]
    
    async def execute(self) -> BotHandle:
        parent = await BotRegistry.get(self.parent_bot_id)
        
        # Create specialist bot
        specialist = await manybot.create_bot(
            name=f"{parent.name}_{self.specialty}",
            goal=f"Handle {self.specialty} for {parent.name}",
            success_criteria=f"All {self.specialty} tasks completed"
        )
        
        # Delegate responsibility
        await manybot.assign_responsibility(
            bot_id=specialist.bot_id,
            paths=self.delegated_paths,
            level="maintainer"
        )
        
        # Update parent's delegation list
        await parent.add_delegate(specialist.bot_id)
        
        return BotHandle(bot_id=specialist.bot_id)

# Usage
specialist = await manybot.spawn_specialist(
    parent_bot_id="api-bot",
    specialty="graphql",
    delegated_paths=["src/api/graphql/**/*.py"]
)
```

## CLI Usage

```bash
# Bot management
manybot create test-guardian \
  --goal "Achieve 90% test coverage" \
  --success "coverage >= 90% AND all tests pass"

manybot assign test-guardian \
  --owns "src/" "tests/" \
  --level owner

manybot status test-guardian
manybot list --active

# Coordination
manybot meeting schedule \
  --topic "API breaking changes" \
  --attendees api-bot,test-bot,docs-bot

manybot meeting list --upcoming
manybot meeting notes meeting-123

# Responsibility queries
manybot owns src/api/  # Shows which bot owns this path
manybot owned-by api-bot  # Shows what api-bot owns

# Delegation
manybot spawn specialist \
  --parent api-bot \
  --specialty "graphql" \
  --paths "src/api/graphql/"

# Bot communication
manybot request \
  --from test-bot \
  --to api-bot \
  --task "Need API mocks for new tests"
```

## Integration Points

- **Remotebot**: Deploys bots as distributed goalbots
- **Goalbot**: Each manybot runs a goalbot for autonomous execution
- **Codebot**: Uses codebot for all file operations
- **GitHub**: Responds to repository events for coordination

## Self-Directed Bot Operations

Manybots are self-directed - they can invoke manybot commands themselves to:
- **Schedule meetings** when coordination is needed
- **Spawn specialists** when they identify work beyond their expertise  
- **Request help** from other bots
- **Delegate responsibilities** when overwhelmed
- **Update their own goals** as they learn and adapt
- **Propose bot mergers** when goals overlap

All structural changes (new bots, goal updates, responsibility changes) are submitted as PRs for human review.

```python
class ManybotSelfDirection:
    """Manybot's ability to use manybot commands"""
    
    async def request_coordination(self, topic: str, with_bots: List[str]):
        """Bot schedules its own meeting when needed"""
        # Bot realizes it needs to coordinate
        meeting = await manybot.schedule_meeting(
            topic=topic,
            attendees=[self.bot_id] + with_bots,
            decisions_needed=self._identify_decisions_needed(),
            context={"requester": self.bot_id}
        )
        return meeting
    
    async def create_specialist(self, specialty: str, reason: str):
        """Bot requests a specialist when identifying a need"""
        # Creates PR for new bot creation with resource requirements
        pr_data = {
            "type": "create_bot",
            "parent": self.bot_id,
            "specialty": specialty,
            "reason": reason,
            "delegated_paths": self._identify_paths_to_delegate(specialty),
            "resource_request": {
                "estimated_hours_per_week": self._estimate_workload(specialty),
                "deployment": "local" if self._is_small_scope(specialty) else "remote"
            }
        }
        
        # PR will be reviewed for both code changes AND budget approval
        pr_url = await self._submit_bot_change_pr(
            title=f"Request {specialty} specialist bot",
            changes=pr_data,
            requires_budget_approval=pr_data["resource_request"]["deployment"] == "remote"
        )
        return pr_url
    
    async def update_goal(self, new_goal: Goal, rationale: str):
        """Bot updates its own goal based on learning"""
        # Submit PR for goal change
        pr_data = {
            "type": "update_goal",
            "bot_id": self.bot_id,
            "old_goal": self.goal.model_dump(),
            "new_goal": new_goal.model_dump(),
            "rationale": rationale
        }
        
        pr_url = await self._submit_bot_change_pr(
            title=f"Update goal for {self.name}",
            changes=pr_data
        )
        return pr_url
    
    async def request_help(self, task: str, suggested_bot: Optional[str] = None):
        """Bot asks for help from another bot"""
        if suggested_bot:
            # Direct request
            await manybot.request(
                from_bot=self.bot_id,
                to_bot=suggested_bot,
                task=task
            )
        else:
            # Broadcast to team
            await manybot.broadcast_request(
                from_bot=self.bot_id,
                task=task,
                required_skills=self._identify_required_skills(task)
            )
```

## Manybot Work Cycle Implementation

```python
class Manybot:
    """Self-directed agent with goals and responsibilities"""
    
    def __init__(self, name: str, goal: Goal, responsibilities: List[Responsibility]):
        self.name = name
        self.goal = goal
        self.responsibilities = responsibilities
        self.state = BotState(
            bot_id=f"{name}_{uuid4().hex[:8]}",
            name=name,
            status="active",
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        # Import and use goalbot for goal execution
        from goalbot import GoalExecutor
        self.goal_executor = GoalExecutor(goal)
        
        # Enable self-direction
        self.self_direction = ManybotSelfDirection(self)
        
        # PydanticAI agent for high-level planning
        self.planner = Agent(
            result_type=WorkPlan,
            system_prompt=self._build_planner_prompt()
        )
    
    async def do_work(self) -> WorkCycleRecord:
        """Execute one autonomous work cycle"""
        
        # 1. Evaluate current state
        goal_progress = self.goal.evaluate()
        
        # 2. Check responsibilities
        responsibility_status = await self._check_responsibilities()
        
        # 3. Plan work based on goal and responsibilities
        plan_context = {
            "goal_progress": goal_progress,
            "responsibilities": responsibility_status,
            "recent_events": await self._get_recent_events(),
            "blocked_tasks": self.state.blocked_tasks
        }
        
        plan_result = await self.planner.run(plan_context)
        work_plan = plan_result.data
        
        # 4. Check if coordination is needed
        if work_plan.coordination_needed:
            # Bot decides to schedule a meeting
            await self.self_direction.request_coordination(
                topic=work_plan.coordination_topic,
                with_bots=work_plan.coordination_needed
            )
        
        # 5. Check if specialist is needed
        if work_plan.specialist_needed:
            # Bot spawns a specialist
            specialist = await self.self_direction.create_specialist(
                specialty=work_plan.specialist_type,
                reason=work_plan.specialist_reason
            )
            # Delegate some tasks to the new specialist
            work_plan.delegate_tasks_to(specialist.bot_id)
        
        # 6. Execute plan using goalbot
        if work_plan.goal_tasks:
            # Execute goal-oriented tasks via goalbot
            goal_result = await self.goal_executor.work_on_tasks(
                tasks=work_plan.goal_tasks,
                max_iterations=work_plan.max_iterations
            )
            
            files_changed = goal_result.files_changed
            commands_executed = goal_result.commands_executed
        
        # 7. Handle responsibilities (code review, maintenance)
        if work_plan.responsibility_tasks:
            for task in work_plan.responsibility_tasks:
                if task.type == "review_change":
                    await self._review_external_change(task)
                elif task.type == "maintain_quality":
                    await self._run_maintenance_workflow(task)
                elif task.type == "request_help":
                    # Bot asks another bot for help
                    await self.self_direction.request_help(
                        task=task.description,
                        suggested_bot=task.suggested_helper
                    )
        
        # 6. Record work cycle
        cycle = WorkCycleRecord(
            cycle_id=str(uuid4()),
            timestamp=datetime.now(),
            goal_progress=goal_progress["overall_progress"],
            files_changed=files_changed,
            commands_executed=commands_executed,
            outcome="success" if goal_result.success else "partial",
            blockers=goal_result.blockers
        )
        
        self.state.work_cycles.append(cycle)
        self.state.last_active = datetime.now()
        
        return cycle
    
    async def _review_external_change(self, task: ResponsibilityTask):
        """Review changes made to owned files by others"""
        # Use codebot's review command
        from codebot import execute_command
        
        review_result = await execute_command(
            "review",
            paths=task.affected_files,
            context={"change_author": task.author}
        )
        
        # Create PR comment or issue if problems found
        if review_result.issues_found:
            await self._create_review_feedback(review_result)
    
    def _build_planner_prompt(self) -> str:
        return f"""You are {self.name}, an autonomous development agent.
        
        Your goal: {self.goal.objective}
        Key results: {[kr.description for kr in self.goal.key_results]}
        
        Your responsibilities:
        {chr(10).join(f'- {r.description}: {r.paths}' for r in self.responsibilities)}
        
        Plan work that:
        1. Makes progress toward your goal's key results
        2. Maintains quality in your areas of responsibility
        3. Responds to external changes affecting your code
        4. Coordinates with other bots when needed
        """

class WorkPlan(BaseModel):
    """Plan for a work cycle"""
    goal_tasks: List[Task]  # Tasks toward goal completion
    responsibility_tasks: List[ResponsibilityTask]  # Maintenance tasks
    
    # Self-direction decisions
    coordination_needed: List[str]  # Bot IDs to coordinate with
    coordination_topic: Optional[str] = None
    specialist_needed: bool = False
    specialist_type: Optional[str] = None
    specialist_reason: Optional[str] = None
    help_requests: List[HelpRequest] = Field(default_factory=list)
    
    max_iterations: int = 10
    rationale: str
```

## Architecture Principles

1. **Clear Ownership**: Every file has a responsible bot/human
2. **Structured Communication**: Formal protocols for bot interaction  
3. **Collaborative Goals**: Bots can refine and delegate objectives
4. **Event-Driven**: GitHub events trigger coordinated responses
5. **Self-Organizing**: Teams evolve based on performance and needs
6. **Layered Execution**: Manybot → Goalbot → Codebot → FileChanges
7. **Human Oversight**: All structural changes via reviewed PRs

## Implementation Roadmap

### Phase 1: Bot Foundation
- **Core Models**: `Bot`, `BotState`, `BotHandle`, `WorkCycleRecord`
- **Bot Registry**: `BotRegistry` singleton with `register()`, `get()`, `list()` methods
- **Work Cycle**: `Manybot.do_work()` method integrating `GoalExecutor`
- **CLI Operations**: `CreateBot`, `BotStatus` command classes
- **PR Submission**: `_submit_bot_change_pr()` for structural changes

### Phase 2: Responsibility System
- **Ownership Models**: `Ownership`, `Responsibility`, `ResponsibilityChange`
- **OWNERS Parser**: YAML parsing and `_update_owners_file()` logic
- **Assignment Operation**: `AssignResponsibility` command implementation
- **GitHub Models**: `GitHubPushEvent`, `PREvent`, `EventNotification`
- **Event Coordinator**: `GitHubCoordinator` for webhook handling

### Phase 3: Bot Coordination
- **Meeting Models**: `MeetingAgenda`, `MeetingOutcome`, `ActionItem`
- **Bot Coordinator**: `BotCoordinator.hold_meeting()` with PydanticAI facilitator
- **Schedule Meeting**: `ScheduleMeeting` operation implementation
- **Decision Application**: `_apply_meeting_outcomes()` method
- **Communication Protocol**: Inter-bot messaging via meeting system

### Phase 4: Self-Direction Capabilities
- **Self-Direction API**: `ManybotSelfDirection` class methods
- **Specialist Spawning**: `SpawnSpecialist` operation with resource requests
- **Goal Evolution**: `update_goal()` with PR-based approval
- **Help System**: `request_help()` with direct and broadcast modes
- **Bot Evolution**: `BotEvolution` class for merge and spawn operations

### Phase 5: Production Operations
- **Work Plan Model**: `WorkPlan` with self-direction decisions
- **Planner Agent**: PydanticAI agent for work planning
- **Monitoring**: Bot health metrics and progress tracking
- **Error Recovery**: Blocker detection and help request automation
