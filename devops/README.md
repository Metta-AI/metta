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
./devops/aws/cmd.sh launch --cmd=sandbox --run=sandbox --job-name=<sandbox_name>
```

### Connecting to a Sandbox

To connect to an existing sandbox:

1. First claim it in Asana: [Dev Cluster](https://app.asana.com/1/1209016784099267/project/1209353759349008/task/1210106185904866?focus=true)
2. Then connect using:

```bash
./devops/aws/cmd.sh ssh --job-name=<sandbox_name>
```

### Stopping a Sandbox

To stop a running sandbox:

```bash
./devops/aws/cmd.sh stop <sandbox_name>
```

## Launching Train Jobs

See [skypilot README](./skypilot/README.md).
