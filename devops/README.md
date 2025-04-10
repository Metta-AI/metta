# Metta AI Devops

Scripts for setting up a Metta AI development environment.

## Git Hooks

This project uses Git hooks to enforce code quality standards.

### Setup

To set up the Git hooks, run:

```bash
# Make the setup script executable
chmod +x devops/setup-git-hooks.sh

# Run the setup script
./devops/setup-git-hooks.sh
```

### Available Hooks

- **pre-commit**: Checks Python files with ruff before committing

