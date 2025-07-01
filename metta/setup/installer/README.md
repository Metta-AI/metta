# Metta Installer

Installation scripts for the Metta project.

## Files

- `install.sh` - Main installation script that:
  - Installs uv if not present
  - Runs uv sync to install Python dependencies
  - Optionally adds metta to PATH
  - Creates environment scripts for different shells

- `install_utils.sh` - Utility functions for PATH configuration adapted from the uv installer
  - Shell detection (bash, zsh, fish, sh)
  - PATH modification helpers
  - Environment script creation
  - CI environment support

## Usage

The install.sh script is symlinked to the project root for convenience. Users should run:

```bash
./install.sh              # Interactive installation
./install.sh --add-to-path  # Automatically add to PATH
./install.sh --help         # Show help
```

## Structure

- `bin/metta` - Wrapper script that runs the metta CLI tool
- Environment scripts (`bin/env`, `bin/env.fish`) are generated during installation
- The installer adds `setup/installer/bin` to PATH rather than creating symlinks
