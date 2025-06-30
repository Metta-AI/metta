# Metta Setup System

Modular dependency management for the Metta AI development environment.

## Quick Start

```bash
# Configure (first time only)
./metta.sh configure

# Install everything
./metta.sh install

# Check status
./metta.sh status
```

## Updating these setup scripts

### Update an Existing Dependency

To update an existing component (e.g., change mettascope install steps):

1. Find the component in `devops/setup/components/` (e.g., `mettascope.py`)
2. Update the version or installation logic in the `install()` method

### Add a New Dependency

To add a new tool or service dependency:

1. Create a new file `devops/setup/components/[tool_name].py` and subclass `SetupModule`

2. (Optional) Add to profiles in `config.py`, specifying expected connected accounts if applicable


## Commands

- `./metta.sh configure --profile=external` - Set up as external contributor
- `./metta.sh install nodejs` - Install specific component
- `./metta.sh install --force` - Reinstall all components
- `./metta.sh status` - Show installation and auth status
