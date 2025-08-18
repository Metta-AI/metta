# Metta Setup System

Modular dependency management for the Metta AI development environment.

## Update an Existing Dependency

To update an existing component (e.g., change mettascope install steps):

1. Find the component in `metta/setup/components/` (e.g., `mettascope.py`)
2. Update the version or installation logic in the `install()` method

## Add a New Dependency

To add a new tool or service dependency:

1. Create a new file `metta/setup/components/[tool_name].py` and subclass `SetupModule`

2. (Optional) Add to profiles in `config.py`, specifying expected connected accounts if applicable

## Per-Component Configuration

Components can store and retrieve their own configuration settings. This allows components to remember user preferences and customize their behavior.

### For Component Developers

1. Define configuration options in your `SetupModule` subclass:

```python
class MyComponentSetup(SetupModule):
    install_once: bool = False

    def dependencies(self) -> list[str]:
        """Define components that should be installed before this one"""
        return ['aws']

    def get_configuration_options(self) -> dict[str, tuple[Any, str]]:
        """Define available configuration options."""
        return {
            "install_mode": ("standard", "Installation mode"),
            "verbose": (False, "Enable verbose output"),
        }

    def configure(self) -> None:
        """Interactive configuration for this component."""
        # Implement interactive configuration logic
        mode = prompt_choice(...)
        self.set_setting("install_mode", mode)

    def install(self):
        # Get settings with defaults - only non-default values are stored
        mode = self.get_setting("install_mode", "standard")
```

Key principles:
- All settings must have defaults defined in `get_configuration_options()`
- Only non-default values are written to disk
- Settings are automatically namespaced under `module_settings.<component_name>`
- Installation (`metta install`) should never prompt for configuration
- Components can depend on other components being installed first. This should be specified in the `dependencies` class var
- `install_once` should be used for components for which if `check_installed` is True, `metta install` only calls the component's `install` if `--force` is provided

### For Users

Configure components using the `metta configure` command:

```bash
# Configure a specific component
metta configure githooks

# Run the general setup wizard
metta configure

# Set a profile directly
metta configure --profile=softmax
```

Component settings are stored in `~/.metta/config.yaml` under the `module_settings` section. Only non-default values are saved:

```yaml
module_settings:
  githooks:
    commit_hook_mode: fix  # Only saved because it differs from default "check"
```

### Example: Git Hooks Configuration

The git hooks component supports three commit hook modes:
- `none`: No pre-commit linting
- `check`: Check only, fail if issues found (default)
- `fix`: Auto-fix issues before committing

To configure:
```bash
metta configure githooks
```

Since `check` is the default, it won't appear in your config file unless you change it.
