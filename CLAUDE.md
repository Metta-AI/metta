# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metta AI is a reinforcement learning project focusing on the emergence of cooperation and alignment in multi-agent AI
systems. It creates a model organism for complex multi-agent gridworld environments to study the impact of social
dynamics (like kinship and mate selection) on learning and cooperative behaviors.

The codebase consists of:

- `metta/`: Core Python implementation for agents, maps, RL algorithms, simulation
- `mettagrid/`: C++/Python grid environment implementation
- `mettascope/`: Visualization and replay tools

## Development Environment Setup

```bash
# Initial setup - installs uv, configures metta, and installs components
./install.sh

# After installation, you can use metta commands directly:
metta status                         # Check component status
metta configure --profile=softmax    # Reconfigure for different profile
metta install aws wandb              # Install specific components

# Run `metta -h` to see all available commands
```

## Common Commands

### Code Quality

```bash
# Run all tests with coverage
metta test --cov=mettagrid --cov-report=term-missing

# Run linting and formatting on python files with Ruff
metta lint # optional --fix and --staged arguments

# Auto-fix Ruff errors with Claude (requires ANTHROPIC_API_KEY)
uv run ./devops/tools/auto_ruff_fix.py path/to/file

# Format shell scripts
./devops/tools/format_sh.sh
```

### Building

Not needed, just run scripts, they'll work automatically through uv-powered shebangs.

```bash
# Clean debug cmake build artifacts. `metta install` also does this
metta clean
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=metta --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_specific.py

# Run tests in parallel
uv run pytest -n auto

# Run only fast tests (skip slow tests)
uv run pytest -m "not slow"
```

### Quick Test Commands (30-60 seconds total)

```bash
# Set a unique test ID for this testing session
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Test ID: $TEST_ID"

# Basic training (will run indefinitely, terminate with Ctrl+C after ~30 seconds)
uv run ./tools/train.py run=test_$TEST_ID +hardware=macbook trainer.num_workers=2

# Using cursor config (limited to 100k steps)
uv run ./tools/train.py +user=cursor run=cursor_$TEST_ID trainer.num_workers=2

# Run simulations on trained model
uv run ./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints device=cpu

# Analyze results
uv run ./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

## Code Architecture

### Agent System

- Each agent has a policy with action spaces and observation spaces
- Policies are stored in `PolicyStore` and managed by `MettaAgent`
- Agent architecture is designed to be adaptable to new game rules and environments
- Neural components can be mixed and matched via configuration

### Environment System

- Gridworld environments with agents, resources, and interaction rules
- Procedural world generation with customizable configurations
- Various environment types with different dynamics and challenges
- Support for different kinship schemes and mate selection mechanisms

### Training Infrastructure

- Distributed reinforcement learning with multi-GPU support
- Integration with Weights & Biases for experiment tracking
- Scalable architecture for training large-scale multi-agent systems
- Support for curriculum learning and knowledge distillation

### Evaluation System

- Comprehensive suite of intelligence evaluations
- Navigation tasks, maze solving, in-context learning
- Cooperation and competition metrics
- Support for tracking and comparing multiple policies

## Configuration System

The project uses OmegaConf for configuration, with config files organized in `configs/`:

- `agent/`: Agent architecture configurations
- `trainer/`: Training configurations
- `sim/`: Simulation configurations
- `hardware/`: Hardware-specific settings
- `user/`: User-specific configurations

### Using Hydra Configuration

Most tools in `tools/` use Hydra for configuration:

- **Override parameters**: `param=value` sets configuration values
- **Compose configs**: `+group=option` loads from `configs/group/option.yaml`
- **User configs**: `+user=<name>` loads from `configs/user/<name>.yaml`

Example:
```bash
./tools/train.py run=my_experiment +hardware=macbook wandb=off trainer.num_workers=4
```

## Testing Philosophy

- Tests should be independent and idempotent
- Tests should be focused on testing one thing
- Tests should cover edge cases and boundary conditions
- Tests are organized in the `tests/` directory, mirroring the project structure

## Code Style Guidelines

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

## Code Review Criteria

When reviewing code, focus on:

- **Type Safety**: Check for missing type annotations, especially return types
- **API Consistency**: Ensure similar functionality follows the same patterns
- **Performance**: Identify potential bottlenecks or inefficient patterns
- **Maintainability**: Look for code that will be difficult to modify or extend
- **Documentation**: Ensure complex logic is properly documented
- **Testing**: Verify that new functionality has appropriate test coverage

## Project-Specific Patterns

### Environment Properties

- Convert methods to properties where appropriate for better API consistency
- Use `@property` decorator for computed attributes
- Ensure all environment properties follow consistent naming patterns
- Example: `action_names()` → `action_names` (property)

### Policy and Agent Management

- Validate policy types with runtime checking using `policy_as_metta_agent()`
- Use Union types for policies: `Union[MettaAgent, DistributedMettaAgent]`
- Ensure proper type safety for policy handling throughout the system

### Device Management

- Add explicit `torch.device` type hints in trainer and simulation modules
- Be consistent about device placement and movement of tensors

## PR Creation Guidelines

When creating PRs (triggered by @claude open-pr):

### Intelligent Branch Targeting

The workflow automatically determines the appropriate base branch:

- **From PR Comments**: New branches are created from the current PR's branch
- **From Issue Comments**: New branches are created from the main branch
- **Example**: If you comment `@claude open-pr` in PR #657 (branch: `robb/0525-agent-type-changes`), Claude will create
  a new branch based on `robb/0525-agent-type-changes`, not main

### Branch Naming Convention

- Use descriptive branch names with prefixes:
  - `feature/add-type-safety` - New functionality
  - `fix/missing-annotations` - Bug fixes
  - `refactor/method-to-property` - Code improvements
  - `docs/update-readme` - Documentation updates
- Include issue number when applicable: `fix/657-type-safety-improvements`

### Commit Message Format

- Follow conventional commit format: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Be specific about what was changed: `fix: add missing return type annotations to PolicyStore methods`
- Reference issues when applicable: `fix: resolve type safety issues (#657)`

### PR Structure Requirements

- **Title**: Clear, concise description of the change
- **Description**: Must include:
  - **What**: Summary of changes made
  - **Why**: Rationale for the change
  - **Testing**: How the changes were verified
  - **Breaking Changes**: Any API changes that affect existing code
- **Linking**: Reference related issues with "Closes #123", "Fixes #123", or "Addresses #123"

### Implementation Strategy

1. **Analyze**: Understand the request and examine current codebase patterns
2. **Plan**: Create focused, incremental changes rather than large rewrites
3. **Implement**: Make changes following established project patterns
4. **Test**: Ensure all existing tests pass and add new tests if needed
5. **Document**: Update docstrings and comments where necessary
6. **Review**: Self-review the changes for consistency with project standards

### Quality Checklist

Before creating a PR, ensure:

- [ ] All new public methods have return type annotations
- [ ] Code follows the established naming conventions
- [ ] No unnecessary comments that restate obvious code
- [ ] Properties are used instead of simple getter methods
- [ ] Proper error handling is implemented
- [ ] Tests pass locally
- [ ] Code is formatted according to project standards

## Interactive Tools

### Exploration and Debugging

```bash
# Interactive simulation for manual testing and exploration
uv run ./tools/play.py run=my_experiment +hardware=macbook wandb=off

# Interactive play with specific policy
uv run ./tools/play.py run=play_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints +hardware=macbook
```

## Navigation Evaluation Database

### Adding Policies to Evaluation Database

```bash
# Add a policy to the navigation evals database
uv run ./tools/sim.py eval=navigation run=RUN_NAME eval.policy_uri=POLICY_URI +eval_db_uri=wandb://artifacts/navigation_db

# Analyze results with heatmap
uv run ./tools/analyze.py run=analyze +eval_db_uri=wandb://artifacts/navigation_db analyzer.policy_uri=POLICY_URI
```

## Smoke Test Mode

When `+smoke_test=true` is added:

- Training: Verifies wandb metrics structure
- Simulation: Runs limited sims and verifies stats DB structure
- Both use deterministic seeds and settings for reproducibility

## Working with Cursorrules

Important patterns from `.cursorrules`:

- Make changes file by file and give a chance to spot mistakes
- Don't suggest whitespace changes or summarize changes made
- Don't remove unrelated code or functionalities
- Provide all edits in a single chunk for the same file
- Use bash with PATH including /opt/homebrew/bin
- Activate virtual environment before python commands: `source .venv/bin/activate`

## Dependencies and Environment

- Python 3.11.7 is required (enforced in pyproject.toml)
- Project uses `uv` for dependency management
- Main dependencies include PyTorch, Gymnasium, PufferLib, Hydra, WandB
- Development dependencies include pytest, ruff, pyright
- Custom dependencies from workspace: metta-app-backend, metta-mettagrid, metta-common, metta-agent