# Metta Setup System

A modular setup system for configuring the Metta AI development environment.

## Quick Start

```bash
# Configure (first time only)
./metta.sh configure

# Install everything
./metta.sh install

# Check status
./metta.sh status
```

## Adding a New Component

1. Create `metta/setup/components/your_component.py`:

```python
import shutil
from metta.setup.registry import register_module
from metta.setup.components.base import SetupModule

@register_module
class YourComponentSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Brief description of your component"

    def is_applicable(self) -> bool:
        # Return True if this component should be available
        return self.config.is_component_enabled("your_component")

    def check_installed(self) -> bool:
        # Return True if already installed
        return shutil.which("your-tool") is not None

    def install(self) -> None:
        # Installation logic here
        self._run("brew install your-tool")

    # Optional: Check authentication status
    def check_connected_as(self) -> str | None:
        result = self._run("your-tool whoami", check=False)
        return result.stdout.strip() if result.returncode == 0 else None
```

2. Since we're using PEP 420 namespace packages, no __init__.py is needed. The module will be auto-discovered.

3. (Optional) Add to default profiles in `config.py`:
```python
DEFAULT_PROFILES = {
    "external": ["system", "core", "your_component"],  # Add here if needed
    "cloud": ["system", "core", "aws", "your_component"],
    # ... etc
}
```

## How It Works

- **Profiles**: `external`, `cloud`, `softmax` with different defaults
- **Config**: Stored in `~/.metta/config.yaml`
- **Scripts**: Components define their own installation process and can call out to existing install scripts (e.g., `mettascope/install.sh`) via `setup_script_location`

## Key Commands

- `./metta.sh configure --profile=PROFILE` - Configure with a specific profile
- `./metta.sh install COMPONENT` - Install a specific component
- `./metta.sh install --force` - Force reinstall all components
- `./metta.sh status` - Show what's installed and authenticated
