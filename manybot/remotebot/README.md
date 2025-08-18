# Remotebot

```bash
# Provision server with tools
remotebot server provision --tools "git,python,pytest,ruff"

# Submit job to available server
remotebot job submit goalbot --goal "Fix failing tests" --server auto

# Monitor resource usage
remotebot resources status
```

## Remote Execution Infrastructure

Remotebot provides the infrastructure for distributed AI agent execution:

- **Server Management**: Provision and manage agent execution servers
- **Job Distribution**: Route work to servers with required tools
- **Tool Provisioning**: Install and verify agent dependencies
- **Resource Scaling**: Auto-scale infrastructure based on demand
- **Container Deployment**: Isolated, reproducible execution environments

## Service Interface

### Provides
- Server provisioning and lifecycle management
- Job queue and distribution system
- Tool installation and environment setup
- Resource monitoring and auto-scaling
- **Budget controls and resource limits**

### Consumes
- Cloud provider APIs for infrastructure
- Container registries for agent images

## Core Architecture

```python
class AgentServer(BaseModel):
    """Managed agent execution environment"""
    server_id: str
    tools: List[str]  # Installed tools
    status: Literal["provisioning", "ready", "busy", "terminating"]
    job_queue: List[Job] = []
    resources: ResourceSpec
    cost_per_hour: float
    budget_remaining: float
    
class ServerManager:
    """Manages fleet of agent servers with budget controls"""
    
    def __init__(self, budget_config: BudgetConfig):
        self.budget = budget_config
        self.spending_tracker = SpendingTracker()
    
    async def provision_server(self, 
                             tools: List[str],
                             requester: str,
                             justification: str) -> AgentServer:
        """Provision new server with budget approval"""
        
        # 1. Calculate costs
        resources = self._calculate_resources(tools)
        estimated_cost = await self._estimate_cost(resources)
        
        # 2. Check budget
        if not await self.budget.can_afford(estimated_cost):
            raise BudgetExceededError(
                f"Cannot provision server: ${estimated_cost}/hr exceeds budget"
            )
        
        # 3. For expensive resources, require human approval
        if estimated_cost > self.budget.auto_approve_limit:
            approval = await self._request_human_approval(
                requester=requester,
                justification=justification,
                estimated_cost=estimated_cost
            )
            if not approval.approved:
                raise ApprovalDeniedError(approval.reason)
        
        # 4. Launch instance
        instance = await self.cloud_provider.launch_instance(
            type="agent-worker",
            resources=resources
        )
        
        # 5. Track spending
        await self.spending_tracker.record_provision(
            server_id=instance.id,
            hourly_cost=estimated_cost,
            requester=requester
        )
        
        # 6. Register server
        server = AgentServer(
            server_id=instance.id,
            tools=tools,
            status="ready",
            cost_per_hour=estimated_cost,
            budget_remaining=self.budget.get_remaining()
        )
        
        self.servers[server.server_id] = server
        return server

class JobDistributor:
    """Distributes work across agent servers"""
    
    async def submit_job(self, job: Job) -> JobHandle:
        # Find available server with required tools
        server = await self.find_or_provision_server(job.required_tools)
        
        # Queue job
        handle = JobHandle(
            job_id=str(uuid4()),
            server_id=server.server_id,
            status="queued"
        )
        
        await server.enqueue(job, handle)
        return handle
        
    async def find_or_provision_server(self, tools: List[str]) -> AgentServer:
        # Check existing servers
        for server in self.servers:
            if set(tools).issubset(set(server.tools)) and server.status == "ready":
                return server
        
        # Provision new server
        return await self.provision_server(tools)
```

## Job Models

```python
class Job(BaseModel):
    """Unit of work for remote execution"""
    id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:8]}")
    type: Literal["codebot", "goalbot", "workflow", "batch"]
    payload: Dict[str, Any]
    required_tools: List[str]
    priority: int = 1
    timeout_seconds: int = 3600
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    
class JobHandle(BaseModel):
    """Handle for tracking job execution"""
    job_id: str
    server_id: str
    status: Literal["queued", "running", "completed", "failed"]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

class ResourceSpec(BaseModel):
    """Resource requirements for a job"""
    cpu_cores: int = 1
    memory_gb: float = 2.0
    gpu_required: bool = False
    disk_gb: float = 10.0
```

## Job Execution Flow

```python
class RemoteJobExecutor:
    """Executes jobs on remote servers with streaming support"""
    
    def __init__(self):
        self.job_distributor = JobDistributor()
        self.result_streamer = ResultStreamer()
    
    async def execute_goalbot_job(self, goal: Goal, server: AgentServer) -> GoalResult:
        """Execute a goalbot goal on remote server"""
        
        # 1. Package job with full context
        job = Job(
            type="goalbot",
            payload={
                "goal": goal.model_dump(),
                "mode": "claudesdk",
                "max_iterations": 50,
                "checkpoint_interval": 5
            },
            required_tools=["git", "python", "codebot", "pytest"],
            resources=ResourceSpec(
                cpu_cores=2,
                memory_gb=4.0
            )
        )
        
        # 2. Submit to server
        handle = await self.job_distributor.submit_job(job)
        
        # 3. Stream results with progress updates
        async for update in self.result_streamer.stream(handle):
            if update.type == "progress":
                yield ProgressUpdate(
                    tasks_completed=update.data["completed"],
                    tasks_total=update.data["total"],
                    current_task=update.data["current"]
                )
            elif update.type == "task_completed":
                yield TaskUpdate(update.data)
            elif update.type == "checkpoint":
                # Save checkpoint for resumption
                await self._save_checkpoint(handle.job_id, update.data)
            elif update.type == "goal_complete":
                return GoalResult(**update.data)
    
    async def execute_codebot_command(self, 
                                    command: str, 
                                    context: ExecutionContext,
                                    server: Optional[AgentServer] = None) -> CommandOutput:
        """Execute a single codebot command remotely"""
        
        # Auto-select server if not provided
        if not server:
            server = await self.job_distributor.find_or_provision_server(
                tools=self._get_required_tools(command),
                requester=context.metadata.get("bot_id", "human"),
                justification=f"Execute {command} command"
            )
        
        job = Job(
            type="codebot",
            payload={
                "command": command,
                "context": context.model_dump(),
                "mode": "oneshot"  # Or "claudesdk" for autonomous completion
            },
            required_tools=self._get_required_tools(command),
            timeout_seconds=300  # 5 min timeout for single commands
        )
        
        handle = await self.job_distributor.submit_job(job, server)
        result = await self.wait_for_completion(handle)
        
        return CommandOutput(**result.output)
    
    async def execute_workflow(self,
                             workflow_name: str,
                             context: Dict[str, Any],
                             server: Optional[AgentServer] = None) -> WorkflowResult:
        """Execute a codebot workflow remotely"""
        
        job = Job(
            type="workflow",
            payload={
                "workflow": workflow_name,
                "context": context,
                "checkpoint_steps": True
            },
            required_tools=["git", "python", "codebot"],
            timeout_seconds=1800  # 30 min for workflows
        )
        
        handle = await self.job_distributor.submit_job(job)
        
        # Stream workflow step completions
        step_results = {}
        async for update in self.result_streamer.stream(handle):
            if update.type == "step_completed":
                step_data = update.data
                step_results[step_data["step_id"]] = step_data["result"]
                yield StepUpdate(step_data)
            elif update.type == "workflow_complete":
                return WorkflowResult(
                    workflow=workflow_name,
                    step_results=step_results,
                    success=update.data["success"]
                )
```

## Job Distribution & Scheduling

```python
class JobQueue:
    """Distribute jobs across available agents"""
    
    def __init__(self):
        self.pending_jobs: asyncio.Queue[Job] = asyncio.Queue()
        self.active_jobs: Dict[str, Job] = {}
        self.server_pool: ServerPool = ServerPool()
    
    async def submit_job(self, job: Job) -> str:
        """Submit job to queue"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job.id = job_id
        await self.pending_jobs.put(job)
        return job_id
    
    async def distribute_jobs(self):
        """Main distribution loop"""
        while True:
            # Get next job
            job = await self.pending_jobs.get()
            
            # Find available server
            server = await self.server_pool.get_available_server(
                requirements=job.requirements
            )
            
            # Deploy and execute
            await self._execute_on_server(job, server)

class JobScheduler:
    """Schedule jobs based on priorities and resources"""
    
    def __init__(self, job_queue: JobQueue):
        self.job_queue = job_queue
        self.resource_monitor = ResourceMonitor()
    
    async def schedule(self, 
                      job_request: JobRequest) -> ScheduleResult:
        """Schedule job with resource optimization"""
        # Check resource availability
        available = await self.resource_monitor.get_available_resources()
        
        # Determine optimal placement
        placement = self._optimize_placement(
            job_request.requirements,
            available
        )
        
        # Create job with placement
        job = Job(
            command=job_request.command,
            context=job_request.context,
            server_affinity=placement.server_id,
            priority=job_request.priority
        )
        
        # Submit to queue
        job_id = await self.job_queue.submit_job(job)
        
        return ScheduleResult(job_id=job_id, 
                            estimated_start=placement.estimated_start)
```

## Container Deployment

```python
class ContainerDeployer:
    """Deploy agents in containers"""
    
    async def deploy_agent(self, 
                          agent_type: str,
                          mode: ExecutionMode,
                          tools: List[str]) -> DeploymentInfo:
        """Deploy containerized agent"""
        # Build image with tools
        image = await self._build_image(agent_type, tools)
        
        # Deploy container
        container = await self._run_container(image, mode)
        
        return DeploymentInfo(
            container_id=container.id,
            endpoint=container.endpoint,
            mode=mode
        )
```

## Background Agent Patterns

```python
class BackgroundAgent(RemoteAgent):
    """Long-running background agent"""
    
    async def run_loop(self):
        """Main execution loop"""
        while self.running:
            # Poll for work
            work = await self.get_next_work()
            
            if work:
                # Execute using codebot
                result = await self.execute_remote(
                    work.command,
                    work.context
                )
                
                # Report results
                await self.report_result(work.id, result)
            
            await asyncio.sleep(self.poll_interval)

class BatchAgent(RemoteAgent):
    """One-shot batch execution"""
    
    async def run_batch(self, 
                       batch_config: BatchConfig) -> BatchResult:
        """Execute batch job and exit"""
        results = []
        
        for task in batch_config.tasks:
            result = await self.execute_remote(
                task.command,
                task.context
            )
            results.append(result)
            
        return BatchResult(results=results)
```

## DevOps Integration

```yaml
# .github/workflows/agent-deploy.yml
name: Deploy Agent
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy code review agent
        run: |
          remotebot deploy code-reviewer \
            --env production \
            --mode persistent \
            --config config/agents/code-reviewer.yaml
```

## Monitoring & Observability

```python
class AgentMetrics:
    """Prometheus metrics for agents"""
    
    commands_total = Counter(
        'agent_commands_total',
        'Total commands executed',
        ['agent_id', 'command', 'status']
    )
    
    execution_duration = Histogram(
        'agent_execution_duration_seconds',
        'Command execution duration',
        ['agent_id', 'command']
    )

class AgentLogger:
    """Structured logging for agents"""
    
    def log_execution(self, 
                     agent_id: str,
                     command: str,
                     result: CommandOutput):
        """Log command execution"""
        self.logger.info(
            "command_executed",
            agent_id=agent_id,
            command=command,
            files_changed=len(result.file_changes),
            summary=result.summary
        )
```

## Remote Storage

```python
class RemoteStorage:
    """Store agent state and results"""
    
    async def save_checkpoint(self, 
                            agent_id: str,
                            state: AgentState):
        """Save agent checkpoint"""
        
    async def load_checkpoint(self, 
                            agent_id: str) -> Optional[AgentState]:
        """Load agent checkpoint"""
        
    async def save_result(self, 
                         execution_id: str,
                         result: CommandOutput):
        """Save execution result"""
```

## CLI Usage

```bash
# Server management
remotebot server list
remotebot server provision --tools "git,python,pytest"
remotebot server status server-123
remotebot server terminate server-123

# Job execution
remotebot job submit goalbot \
  --goal "Improve test coverage" \
  --server auto  # Finds or provisions suitable server

remotebot job submit codebot \
  --command "refactor" \
  --paths "src/" \
  --tools "ruff,pytest"

# Job monitoring
remotebot job list --active
remotebot job status job-123
remotebot job logs job-123 --follow
remotebot job cancel job-123

# Resource management
remotebot resources status  # Shows server utilization
remotebot resources scale --min 2 --max 10  # Auto-scaling
```

## Integration Points

- **Codebot**: Uses codebot as the core execution engine
- **Goalbot**: Deploys goalbot agents for distributed goal achievement
- **Manybot**: Provides infrastructure for manybot coordination

## Tool Provisioning

```python
class ToolManager:
    """Ensure agents have required tools"""
    
    async def provision_tools(self, server: Server, tools: List[str]):
        """Install tools on server"""
        for tool in tools:
            await self._install_tool(server, tool)
    
    async def verify_tools(self, server: Server, tools: List[str]) -> bool:
        """Verify all tools are available"""
        for tool in tools:
            if not await self._check_tool(server, tool):
                return False
        return True
```

## Architecture Principles

1. **Location Agnostic**: Agents run anywhere - local, cloud, edge
2. **Resource Aware**: Smart job distribution based on available resources
3. **Budget Controlled**: Resource usage requires approval and tracking
4. **Tool Provisioning**: Automatic setup of required tools and dependencies
5. **Observable**: Built-in metrics, logging, and cost tracking
6. **Resilient**: Checkpointing, retries, and graceful degradation
7. **Scalable**: Horizontal scaling within budget constraints

## Implementation Roadmap

### Phase 1: Job Execution Foundation
- **Job Models**: `Job`, `JobHandle`, `ResourceSpec` with Pydantic validation
- **Job Queue**: `JobQueue` class with async queue and `submit_job()` method
- **Local Executor**: `RemoteJobExecutor` for in-process job execution
- **Result Models**: `CommandOutput`, `GoalResult`, `WorkflowResult` types

### Phase 2: Server Management
- **Server Models**: `AgentServer` with status tracking and job queue
- **Server Manager**: `ServerManager` with budget controls and provisioning
- **Tool Manager**: `ToolManager.provision_tools()` and `verify_tools()` methods
- **Container Deployer**: `ContainerDeployer` for isolated execution environments

### Phase 3: Job Distribution
- **Job Distributor**: `JobDistributor` with `find_or_provision_server()` logic
- **Resource Monitoring**: `ResourceMonitor` for capacity tracking
- **Result Streaming**: `ResultStreamer` with progress updates and checkpoints
- **Job Scheduler**: Priority-based scheduling with resource optimization

### Phase 4: Infrastructure Scaling
- **Budget Controls**: `BudgetConfig`, `SpendingTracker` for cost management
- **Auto-scaling**: Server provisioning based on queue depth and resource needs
- **Metrics Collection**: `AgentMetrics` with Prometheus integration
- **Remote Storage**: `RemoteStorage` for checkpoints and results