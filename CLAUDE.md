# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Guidance - How to Work on Tasks

### Learning User Preferences

**When receiving general directions that might apply to the codebase:**

1. **Identify patterns** - When the user gives instructions that seem like general preferences or patterns that should
   be applied across the codebase
2. **Propose updates** - Suggest adding these preferences to CLAUDE.md with specific text
3. **Ask for confirmation** - Present the proposed update and ask: "Should I add this preference to CLAUDE.md so I
   remember it for future work?"
4. **Apply if approved** - Update CLAUDE.md only after receiving explicit approval

This allows Claude to learn and remember user preferences over time, building a personalized understanding of how to
work with this specific codebase.

### Plan & Review Process

Preferred: Start in plan mode for larger or ambiguous tasks. For small, surgical changes, you may proceed directly with
implementation.

1. **Enter plan mode first** - Use the ExitPlanMode tool only after presenting a complete plan
2. **Create a task plan** - Write your plan to `.claude/tasks/TASK_NAME.md` with:
   - Clear problem statement
   - MVP approach (always think minimal viable solution first)
   - Step-by-step implementation plan
   - Success criteria
3. **Use appropriate tools** - If the task requires external knowledge or complex searches, use the Task tool with
   appropriate agents
4. **Optional review** - If the plan is non-trivial or high-risk, request a quick review before implementing
5. **Proceed when ready** - If low-risk and scoped, you may proceed without explicit approval

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
5. **CRITICAL: Always format Python code** - After editing any Python file (\*.py), immediately run:
   ```bash
   metta lint --fix
   ```
   Or alternatively, for individual files:
   ```bash
   ruff format [file_path]
   ruff check --fix [file_path]
   ```
   Note: Only run these commands on Python files, not on other file types like Markdown, YAML, etc.

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

Metta AI is a reinforcement learning project focusing on the emergence of cooperation and alignment in multi-agent AI
systems. It creates a model organism for complex multi-agent gridworld environments to study the impact of social
dynamics (like kinship and mate selection) on learning and cooperative behaviors.

### Repository Structure

- `metta/`: Core Python implementation for agents, maps, RL algorithms, simulation
- `packages/mettagrid/`: C++/Python grid environment implementation with Pybind11 bindings
- `mettascope/`: Interactive visualization and replay tools (TypeScript/web-based)
- `observatory/`: React-based dashboard for viewing training runs and evaluations
- `gridworks/`: Next.js web interface
- `app_backend/`: FastAPI backend server for stats and data services

### Architecture Overview

#### Agent System

- Each agent has a policy with action spaces and observation spaces
- Policies are stored using `CheckpointManager` and managed by `MettaAgent`
- Agent architecture is designed to be adaptable to new game rules and environments
- Neural components can be mixed and matched via configuration
- Key classes:
  - `agent.src.metta.agent.metta_agent.MettaAgent` - Main agent implementation
  - `metta.rl.checkpoint_manager.CheckpointManager` - Manages policy checkpoints
  - `agent.src.metta.agent.distributed_metta_agent.DistributedMettaAgent` - Multi-GPU agent

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

# If you encounter missing dependencies or import errors (like pufferlib), run:
metta install                        # Reinstall all components
# or
metta install core                   # Reinstall core dependencies only
```

### Key Entry Points

#### Training and Evaluation Pipeline

All tools are now run through `./tools/run.py` with recipe functions:

1. **Training**: Use recipe functions for different training configurations

   ```bash
   # Training with arena recipe
   uv run ./tools/run.py experiments.recipes.arena.train run=my_experiment

   # Training with navigation recipe
   uv run ./tools/run.py experiments.recipes.navigation.train run=my_experiment
   ```

2. **Simulation/Evaluation**: Run evaluation suites on trained policies

```bash
# Run evaluation
uv run ./tools/run.py experiments.recipes.arena.evaluate \
  policy_uri=file://./train_dir/my_experiment/checkpoints/my_experiment:v12.pt

# Using a remote S3 checkpoint
uv run ./tools/run.py experiments.recipes.arena.evaluate \
  policy_uri=s3://my-bucket/checkpoints/my-training-run/my-training-run:v12.pt
```

3. **Analysis**: Analyze evaluation results

   ```bash
   uv run ./tools/run.py experiments.recipes.arena.analyze eval_db_uri=./train_dir/eval/stats.db
   ```

4. **Interactive Play**: Test policies interactively (browser-based)

```bash
uv run ./tools/run.py experiments.recipes.arena.play \
  policy_uri=file://./train_dir/my_experiment/checkpoints/my_experiment:v12.pt
```

5. **View Replays**: Watch recorded gameplay

```bash
uv run ./tools/run.py experiments.recipes.arena.replay \
  policy_uri=s3://my-bucket/checkpoints/local.alice.1/local.alice.1:v10.pt
```

#### Visualization Tools

**Note**: These commands start development servers that run indefinitely. In Claude Code, they may hang without clear
feedback. Consider running them in separate terminals outside of Claude Code.

- **MettaScope**: Run `cd mettascope && pnpm run dev` for interactive replay viewer
- **Observatory**: Run `cd observatory && pnpm run dev` for training dashboard
- **GridWorks**: Run `cd gridworks && pnpm run dev` for web interface

### Common Commands

See @.cursor/commands.md for quick test commands and examples.

#### Code Quality

```bash
# Run all tests with coverage
metta test --cov=mettagrid --cov-report=term-missing

# Run specific test modules
metta test tests/rl/test_trainer_config.py -v
metta test tests/sim/ -v

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
# Clean Bazel build artifacts. `metta install` also does this
metta clean
```

### Configuration System

The project uses OmegaConf for configuration, with config files organized in `configs/`:

- `agent/`: Agent architecture configurations (latent_attn_tiny, latent_attn_small, latent_attn_med, fast,
  reference_design)
- `trainer/`: Training configurations
- `sim/`: Simulation configurations (navigation, memory, arena, etc.)
- `user/`: User-specific configurations
- `wandb/`: Weights & Biases settings

#### Running Training and Tools

All tools are now run through `./tools/run.py` with recipe functions:

```bash
# Training with arena recipe
uv run ./tools/run.py experiments.recipes.arena.train run=my_experiment

# Training with navigation recipe
uv run ./tools/run.py experiments.recipes.navigation.train run=my_experiment

# Play/test a trained policy (interactive browser)
uv run ./tools/run.py experiments.recipes.arena.play \
  policy_uri=file://./train_dir/my_experiment/checkpoints/my_experiment:v12.pt

# Run evaluation
uv run ./tools/run.py experiments.recipes.arena.evaluate \
  policy_uri=file://./train_dir/my_experiment/checkpoints/my_experiment:v12.pt

# View replays
uv run ./tools/run.py experiments.recipes.arena.replay \
  policy_uri=s3://my-bucket/checkpoints/local.alice.1/local.alice.1:v10.pt
```

#### Configuration System

The project now uses Pydantic-based configuration instead of Hydra/YAML. Configurations are built programmatically in
recipe files:

- Recipes define training setups, environments, and evaluation suites
- Each recipe function returns a Tool configuration object
- Override parameters can be passed via command line arguments to the recipe functions

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

1. Use smaller batch sizes for debugging
2. Check wandb logs for metrics anomalies
3. Use `./tools/run.py experiments.recipes.arena.play` for interactive debugging (Note: Less useful in Claude Code due
   to interactive nature)

#### Performance Profiling

1. Use `torch.profiler` integration in trainer
2. Monitor GPU utilization:
   - On NVIDIA GPUs: `nvidia-smi`
   - On macOS: Use Activity Monitor or `sudo powermetrics --samplers gpu_power`
3. Check environment step timing in vecenv
4. Profile C++ code with Bazel debug builds

---

## Code Standards

### Code Style Guidelines

- **Import placement**: All imports MUST be at the top of the file, never inside functions or methods
  - Follow PEP 8 import ordering: standard library, third-party, local imports
  - Each group separated by a blank line
  - No imports inside function bodies, even for lazy loading
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
- **Docstring style**: Use concise docstrings without Args: and Returns: blocks. The function signature and type hints
  provide parameter information; docstrings should focus on purpose and behavior
- **Multi-line docstring format**: Start with `"""` followed immediately by text on the same line, end with `"""` on its
  own line

### Class Member Naming Conventions

- **Private members**: All class attributes and methods that are internal implementation details MUST start with
  underscore (`_`)
  - Example: `self._internal_state`, `def _process_data(self):`
- **Public members**: Only expose class members without underscore if they are part of the public API
  - Example: `self.name` (if users should access it), `def process(self):` (if users should call it)
- **Protected members**: Use single underscore for "protected" members that subclasses might need
- **Name mangling**: Use double underscore (`__`) sparingly, only when you need Python's name mangling to avoid subclass
  conflicts
- **Test access exception**: Tests are allowed to access private members (those starting with `_`) for thorough testing
  - This allows tests to verify internal state and implementation details
  - Tests can directly access `_private_method()` or `self._private_attribute`
  - Rationale: While private members indicate "internal use", tests need deep access to verify correctness
- **Properties**: Use `@property` decorator to expose computed values or controlled access to private attributes

  ```python
  class Example:
      def __init__(self):
          self._value = 0  # Private attribute

      @property
      def value(self):  # Public property
          return self._value
  ```

### Project-Specific Patterns

#### Environment Properties

- Convert methods to properties where appropriate for better API consistency
- Use `@property` decorator for computed attributes
- Ensure all environment properties follow consistent naming patterns
- Example: `action_names()` → `action_names` (property)

#### Policy and Agent Management

- Validate policy types with runtime checking
- Use Union types for policies: `Union[MettaAgent, DistributedMettaAgent]`
- Ensure proper type safety for policy handling throughout the system
- Policy filenames embed the run: `file://path/to/run/checkpoints/<run_name>:v{epoch}.pt` or
  `s3://bucket/path/run/checkpoints/<run_name>:v{epoch}.pt`

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
- **Always use `uv run` for testing Python files** - This ensures proper environment activation and dependency
  resolution
- Test organization:
  - `tests/rl/` - Reinforcement learning components
  - `tests/sim/` - Simulation and evaluation
  - `tests/map/` - Map generation and scene loading
  - `tests/sweep/` - Hyperparameter sweep infrastructure
  - `packages/mettagrid/tests` - Environment-specific tests

### Code Review Criteria

When reviewing code, focus on:

- **Type Safety**: Check for missing type annotations, especially return types
- **API Consistency**: Ensure similar functionality follows the same patterns
- **Performance**: Identify potential bottlenecks or inefficient patterns
- **Maintainability**: Look for code that will be difficult to modify or extend
- **Documentation**: Ensure complex logic is properly documented
- **Testing**: Verify that new functionality has appropriate test coverage
- **Conciseness**: Less code is almost always preferred - avoid unnecessary complexity
- **Professional Tone**: Avoid emojis in code, comments, and commit messages

---

## PR & Collaboration Guidelines

### PR Creation (triggered by @claude open-pr)

#### Intelligent Branch Targeting

The workflow automatically determines the appropriate base branch:

- **From PR Comments**: New branches are created from the current PR's branch
- **From Issue Comments**: New branches are created from the main branch
- **Example**: If you comment `@claude open-pr` in PR #657 (branch: `robb/0525-agent-type-changes`), Claude will create
  a new branch based on `robb/0525-agent-type-changes`, not main

#### Branch Naming Convention

- Use descriptive branch names with prefixes:
  - `feature/add-type-safety` - New functionality
  - `fix/missing-annotations` - Bug fixes
  - `refactor/method-to-property` - Code improvements
  - `docs/update-readme` - Documentation updates
- Include issue number when applicable: `fix/657-type-safety-improvements`

#### Commit Message Format

- Follow conventional commit format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Be specific about what was changed: `fix: add missing return type annotations to CheckpointManager methods`
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

- use uv run <cmd> instead of python
