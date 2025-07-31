# Codebot

## Immediate vision

```bash
# Fix failing tests (reads error from clipboard)
codebot debug-tests metta/rl

# Launches a claude code with a subagent prompt + additional context like `git diff main`
codebot debug-tests metta/rl -i


# Fix ruff issues
codebot lint

```

## Documentation

### Stage 1: Codebot CLI

Start here - immediate value with simple commands and interactive modes.

- [CODEBOT.md](CODEBOT.md) - Core CLI, commands, and interactive modes

### Stage 2: Workflows

Chain commands together for complex multi-step tasks.

- [WORKFLOWS.md](WORKFLOWS.md) - Composing commands into workflows

### Stage 3: Manybots

Autonomous agents that work toward goals with clear ownership.

- [MANYBOT.md](MANYBOT.md) - Goal-driven agents with areas of responsibility

## Examples

### Stage 1: Direct Commands

```bash
# One-shot commands
codebot test src/api.py
codebot debug-tests      # reads from clipboard
codebot review          # reviews git diff

# Interactive modes
codebot refactor -i     # Claude Code
codebot fix -p          # Pipeline conversation
codebot lint -r         # Review each change
```

### Stage 2: Workflows

```bash
# Built-in workflows
codebot tdd src/feature.py
codebot feature -m "Add user notifications"

# Custom workflow
codebot db-migration
```

### Stage 3: Autonomous Agents

```bash
# Create goal-driven bots
manybot create test-bot \
  --goal "90% test coverage by Q1" \
  --owns "src/" "tests/"

# Manage bots
manybot list
manybot status test-bot
```
