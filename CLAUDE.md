# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Guidance - How to Work on Tasks

### Plan & Review Process

**IMPORTANT: Always start in plan mode before implementing any changes.**

1. **Enter plan mode first** - Use the ExitPlanMode tool only after presenting a complete plan
2. **Create a task plan** - Write your plan to `.claude/tasks/TASK_NAME.md` with:
   - Clear problem statement
   - MVP approach (always think minimal viable solution first)
   - Step-by-step implementation plan
   - Success criteria
3. **Use appropriate tools** - If the task requires external knowledge or complex searches, use the Task tool with appropriate agents
4. **Request review** - After writing the plan, explicitly ask: "Please review this plan before I proceed with implementation"
5. **Wait for approval** - Only exit plan mode and begin implementation after receiving approval

#### Plan Template (.claude/tasks/TASK_NAME.md)

```markdown
# Task: [TASK_NAME]

## Problem Statement
[Clear description of what needs to be done]

## MVP Approach
[Minimal solution that solves the core problem]

## Implementation Plan
1. [Step 1]
2. [Step 2]
3. ...

## Success Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] ...

## Implementation Updates
[This section will be updated during implementation]
```

### During Implementation

**Maintain the plan as living documentation throughout implementation:**

1. **Update as you work** - When you discover new information or need to adjust the approach, update the plan file
2. **Document completed steps** - After completing each major step, append a brief description:
   ```markdown
   ### Step 1 Complete: [Date/Time]
   - Changed: [what was changed]
   - Files affected: [list files]
   - Key decisions: [any important choices made]
   ```
3. **Track deviations** - If you need to deviate from the plan, document why and update the approach
4. **Keep it concise** - Focus on what changed and why, not how (the code shows how)

### After Implementation

1. **Final update** - Update the task file with:
   - Summary of what was accomplished
   - Any known limitations or future work
   - Lessons learned (if applicable)
2. **Verify success criteria** - Check off completed criteria in the plan
3. **Clean up** - Ensure all code is properly tested and documented

---

## Prompt Engineering Principles

### Core Principles

- **Extract & Reframe** - Identify user's true intent and convert to clear, targeted prompts
- **Optimize for LLM** - Structure inputs to enhance reasoning, formatting, and creativity
- **Handle Edge Cases** - Anticipate ambiguities and clarify them proactively
- **Domain Expertise** - Use appropriate terminology, constraints, and examples
- **Modular Design** - Create reusable, adaptable prompt templates

### Design Protocol

1. **Define Objective** - Clear, unambiguous outcomes
2. **Understand Domain** - Tailor language to specific context
3. **Choose Format** - Match output format to use case (JSON, markdown, code, etc.)
4. **Set Constraints** - Specify limits (length, tone, structure)
5. **Provide Examples** - Use few-shot learning when helpful
6. **Test & Refine** - Predict responses and iterate

### Guiding Question

"Would this prompt produce the best result for a non-expert user?"

The goal: Design interactions, not just instructions.

---

## Project Overview

### What is Metta AI?

Metta AI is a reinforcement learning project focusing on the emergence of cooperation and alignment in multi-agent AI systems. It creates a model organism for complex multi-agent gridworld environments to study the impact of social dynamics (like kinship and mate selection) on learning and cooperative behaviors.

### Repository Structure

- `metta/`: Core Python implementation for agents, maps, RL algorithms, simulation
- `mettagrid/`: C++/Python grid environment implementation with Pybind11 bindings
- `mettascope/`: Interactive visualization and replay tools (TypeScript/web-based)
- `observatory/`: React-based dashboard for viewing training runs and evaluations
- `gridworks/`: Next.js web interface
- `app_backend/`: FastAPI backend server for stats and data services

### Architecture Overview

#### Agent System
- Each agent has a policy with action spaces and observation spaces
- Policies are stored in `PolicyStore` and managed by `MettaAgent`
- Agent architecture is designed to be adaptable to new game rules and environments
- Neural components can be mixed and matched via configuration
- Key classes:
  - `metta.agent.metta_agent.MettaAgent` - Main agent implementation
  - `metta.agent.policy_store.PolicyStore` - Manages policy checkpoints
  - `metta.agent.distributed_metta_agent.DistributedMettaAgent` - Multi-GPU agent

#### Environment System
- Gridworld environments with agents, resources, and interaction rules
- Procedural world generation with customizable configurations
- Various environment types with different dynamics and challenges
- Support for different kinship schemes and mate selection mechanisms
- Key components:
  - `mettagrid/` - C++ core implementation for performance
  - `metta.map.mapgen` - Procedural map generation
  - `metta.map.scene` - Scene configuration and loading

#### Training Infrastructure
- Distributed reinforcement learning with multi-GPU support
- Integration with Weights & Biases for experiment tracking
- Scalable architecture for training large-scale multi-agent systems
- Support for curriculum learning and knowledge distillation
- Key components:
  - `metta.rl.trainer.Trainer` - Main training loop
  - `metta.rl.vecenv` - Vectorized environment wrapper
  - `metta.rl.kickstarter` - Policy initialization strategies

#### Evaluation System
- Comprehensive suite of intelligence evaluations
- Navigation tasks, maze solving, in-context learning
- Cooperation and competition metrics
- Support for tracking and comparing multiple policies
- Key components:
  - `metta.sim.simulation.Simulation` - Core simulation engine
  - `metta.sim.simulation_suite` - Evaluation task suites
  - `metta.eval.eval_stats_db` - SQLite-based stats storage

---

## Development Guide

### Environment Setup

```bash
# Initial setup - installs uv, configures metta, and installs components
./install.sh

# After installation, you can use metta commands directly:
metta status                         # Check component status
metta configure --profile=softmax    # Reconfigure for different profile
metta install aws wandb              # Install specific components

# Run `metta -h` to see all available commands
```

### Key Entry Points

#### Training and Evaluation Pipeline

1. **Training**: `tools/train.py` - Main training script using Hydra configuration
   ```bash
   uv run ./tools/train.py run=my_experiment +hardware=macbook
   ```

2. **Simulation/Evaluation**: `tools/sim.py` - Run evaluation suites on trained policies
   ```bash
   uv run ./tools/sim.py run=eval policy_uri=file://./checkpoints/policy.pt
   ```

3. **Analysis**: `tools/analyze.py` - Analyze evaluation results and generate reports
   ```bash
   uv run ./tools/analyze.py run=analysis analysis.eval_db_uri=./train_dir/eval/stats.db
   ```

4. **Interactive Play**: `tools/play.py` - Manual testing and exploration
   ```bash
   uv run ./tools/play.py run=play +hardware=macbook
   ```

5. **Sweep Management**: `tools/sweep_setup.py`, `tools/sweep_prepare_run.py` - Hyperparameter sweep tools

#### Visualization Tools

- **MettaScope**: Run `cd mettascope && npm run dev` for interactive replay viewer
- **Observatory**: Run `cd observatory && npm run dev` for training dashboard
- **GridWorks**: Run `cd gridworks && npm run dev` for web interface

### Common Commands

See @.cursor/commands.md for quick test commands and examples.

#### Code Quality

```bash
# Run all tests with coverage
metta test --cov=mettagrid --cov-report=term-missing

# Run specific test modules
uv run pytest tests/rl/test_trainer_config.py -v
uv run pytest tests/sim/ -v

# Run linting and formatting on python files with Ruff
metta lint # optional --fix and --staged arguments

# Auto-fix Ruff errors with Claude (requires ANTHROPIC_API_KEY)
uv run ./devops/tools/auto_ruff_fix.py path/to/file

# Format shell scripts
./devops/tools/format_sh.sh
```

#### Building

Not needed, just run scripts, they'll work automatically through uv-powered shebangs.

```bash
# Clean debug cmake build artifacts. `metta install` also does this
metta clean
```

### Configuration System

The project uses OmegaConf for configuration, with config files organized in `configs/`:

- `agent/`: Agent architecture configurations (tiny, small, medium, reference_design)
- `trainer/`: Training configurations
- `sim/`: Simulation configurations (navigation, memory, arena, etc.)
- `hardware/`: Hardware-specific settings (macbook, github)
- `user/`: User-specific configurations
- `wandb/`: Weights & Biases settings

#### Configuration Override Examples

```bash
# Override specific parameters
uv run ./tools/train.py trainer.num_workers=4 trainer.total_timesteps=100000

# Use different agent architecture
uv run ./tools/train.py agent=latent_attn_tiny

# Disable wandb
uv run ./tools/train.py wandb=off
```

#### Hydra Configuration Patterns

- Use `+` prefix to add new config groups: `+hardware=macbook`
- Use `~` prefix to override without schema validation: `~trainer.num_workers=2`
- Use `++` prefix to force override: `++trainer.device=cpu`
- Config composition order matters - later overrides take precedence

### Development Workflows

#### Adding a New Evaluation Task

1. Create new config in `configs/sim/`
2. Implement evaluation logic in `metta/sim/`
3. Add tests in `tests/sim/`
4. Register with simulation suite if needed

#### Modifying Agent Architecture

1. Update or create config in `configs/agent/`
2. Modify neural network components in `metta/agent/`
3. Ensure compatibility with existing training pipeline
4. Test with small-scale training run

#### Debugging Training Issues

1. Enable debug logging: `HYDRA_FULL_ERROR=1`
2. Use smaller batch sizes for debugging
3. Check wandb logs for metrics anomalies
4. Use `tools/play.py` for interactive debugging

#### Performance Profiling

1. Use `torch.profiler` integration in trainer
2. Monitor GPU utilization with `nvidia-smi`
3. Check environment step timing in vecenv
4. Profile C++ code with cmake debug builds

---

## Code Standards

### Code Style Guidelines

- Use modern Python typing syntax (PEP 585: `list[str]` instead of `List[str]`)
- Use Union type syntax for Python 3.10+ (`type | None` instead of `Optional[type]`)
- Follow selective type annotation guidelines:
  - **Always annotate**: All function parameters
  - **Selectively annotate returns for**:
    - Public API functions/methods (not prefixed with \_)
    - Functions with complex logic or multiple branches
    - Functions where the return type isn't obvious from the name
    - Functions that might return None in some cases
  - **Skip return annotations for**:
    - Private methods internal to a class
    - Functions enclosed within other functions
    - Simple getters/setters with obvious returns
    - Very short functions (1-3 lines) with obvious returns
  - **Variable annotations**: Only when type inference fails or for empty collections
- Prefer dataclasses over TypedDict for complex data structures
- Use descriptive variable names that clearly indicate purpose
- Remove unnecessary comments that just restate what the code does
- Prefer properties over methods for computed attributes using `@property` decorator
- Implement proper error handling with clear, actionable error messages

### Project-Specific Patterns

#### Environment Properties
- Convert methods to properties where appropriate for better API consistency
- Use `@property` decorator for computed attributes
- Ensure all environment properties follow consistent naming patterns
- Example: `action_names()` → `action_names` (property)

#### Policy and Agent Management
- Validate policy types with runtime checking using `policy_as_metta_agent()`
- Use Union types for policies: `Union[MettaAgent, DistributedMettaAgent]`
- Ensure proper type safety for policy handling throughout the system
- Policy URIs follow format: `file://path/to/checkpoint` or `wandb://project/run/artifact`

#### Device Management
- Add explicit `torch.device` type hints in trainer and simulation modules
- Be consistent about device placement and movement of tensors
- Use `device=cpu` on macOS (no CUDA support)

### Testing Philosophy

See @.cursor/docs.md for testing examples and quick test commands.

- Tests should be independent and idempotent
- Tests should be focused on testing one thing
- Tests should cover edge cases and boundary conditions
- Tests are organized in the `tests/` directory, mirroring the project structure
- Test organization:
  - `tests/rl/` - Reinforcement learning components
  - `tests/sim/` - Simulation and evaluation
  - `tests/map/` - Map generation and scene loading
  - `tests/sweep/` - Hyperparameter sweep infrastructure
  - `tests/mettagrid/` - Environment-specific tests

### Code Review Criteria

When reviewing code, focus on:

- **Type Safety**: Check for missing type annotations, especially return types
- **API Consistency**: Ensure similar functionality follows the same patterns
- **Performance**: Identify potential bottlenecks or inefficient patterns
- **Maintainability**: Look for code that will be difficult to modify or extend
- **Documentation**: Ensure complex logic is properly documented
- **Testing**: Verify that new functionality has appropriate test coverage

---

## PR & Collaboration Guidelines

### PR Creation (triggered by @claude open-pr)

#### Intelligent Branch Targeting

The workflow automatically determines the appropriate base branch:

- **From PR Comments**: New branches are created from the current PR's branch
- **From Issue Comments**: New branches are created from the main branch
- **Example**: If you comment `@claude open-pr` in PR #657 (branch: `robb/0525-agent-type-changes`), Claude will create a new branch based on `robb/0525-agent-type-changes`, not main

#### Branch Naming Convention

- Use descriptive branch names with prefixes:
  - `feature/add-type-safety` - New functionality
  - `fix/missing-annotations` - Bug fixes
  - `refactor/method-to-property` - Code improvements
  - `docs/update-readme` - Documentation updates
- Include issue number when applicable: `fix/657-type-safety-improvements`

#### Commit Message Format

- Follow conventional commit format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Be specific about what was changed: `fix: add missing return type annotations to PolicyStore methods`
- Reference issues when applicable: `fix: resolve type safety issues (#657)`

#### PR Structure Requirements

- **Title**: Clear, concise description of the change
- **Description**: Must include:
  - **What**: Summary of changes made
  - **Why**: Rationale for the change
  - **Testing**: How the changes were verified
  - **Breaking Changes**: Any API changes that affect existing code
- **Linking**: Reference related issues with "Closes #123", "Fixes #123", or "Addresses #123"

#### Implementation Strategy

1. **Analyze**: Understand the request and examine current codebase patterns
2. **Plan**: Create focused, incremental changes rather than large rewrites
3. **Implement**: Make changes following established project patterns
4. **Test**: Ensure all existing tests pass and add new tests if needed
5. **Document**: Update docstrings and comments where necessary
6. **Review**: Self-review the changes for consistency with project standards

#### Quality Checklist

Before creating a PR, ensure:

- [ ] All new public methods have return type annotations
- [ ] Code follows the established naming conventions
- [ ] No unnecessary comments that restate obvious code
- [ ] Properties are used instead of simple getter methods
- [ ] Proper error handling is implemented
- [ ] Tests pass locally
- [ ] Code is formatted according to project standards