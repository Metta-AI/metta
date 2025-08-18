# OpenHands Setup for Metta AI

This directory contains setup scripts that automatically configure the Metta AI development environment when starting an
OpenHands conversation.

## Files

- `setup.sh` - Main setup script that runs automatically when OpenHands starts
- `README.md` - This documentation file

## What the setup script does

The `setup.sh` script performs the following actions:

1. **Configures git** with OpenHands defaults
2. **Installs uv package manager** if not already present
3. **Runs the official setup script** (`devops/setup_dev.sh`) which:
   - Installs all Python dependencies using `uv sync`
   - Builds C++ extensions required by Metta
   - Performs comprehensive sanity checks
   - Sets up the complete development environment
4. **Verifies installation** by testing core module imports
5. **Displays helpful information** about available commands

The script leverages the existing `devops/setup_dev.sh` infrastructure rather than duplicating setup logic.

## Manual setup

If you need to run the setup manually:

```bash
./.openhands/setup.sh
```

## Troubleshooting

If the setup fails:

1. Check that you're in the Metta project root directory
2. Ensure you have internet access for downloading dependencies
3. For C++ build issues, you may need to install build tools
4. Check the Python version (requires 3.11.7)

## Environment

After setup, use `uv run` to execute commands in the project environment:

```bash
# Run Python scripts
uv run python -c "import metta; print('Metta is ready!')"

# Run training
uv run ./tools/train.py run=my_experiment wandb=off

# Run tests
uv run pytest

# Format code
uv run ruff format && uv run ruff check
```

## Project Structure

The Metta AI project includes:

- **metta/** - Core AI framework
- **mettagrid/** - Grid world environment
- **tools/** - Training and evaluation scripts
- **tests/** - Test suite
- **devops/** - Development and deployment tools
- **docs/** - Documentation

For more information, see the main [README.md](../README.md) in the project root.
