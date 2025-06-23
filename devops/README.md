# Metta AI Devops

Scripts for setting up a Metta AI development environment and launching cloud jobs and sandboxes.

## Initial Setup

To set up a new Mac machine:

```bash
python devops/macos/setup_machine.py
```

## Git Hooks

This project uses Git hooks to enforce code quality standards.

### Setup

To set up the Git hooks, run:

```bash
# Make the setup script executable
chmod +x devops/setup_git_hooks.sh

# Run the setup script
./devops/setup_git_hooks.sh
```

### Available Hooks

- **pre-commit**: Checks Python files with ruff before committing

## Launching Sandbox Environments

To launch a new sandbox on AWS:

```bash
./devops/skypilot/sandbox.py
```

## Launching Train Jobs

See [skypilot README](./skypilot/README.md).
