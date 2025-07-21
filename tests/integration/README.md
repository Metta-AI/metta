# PufferLib Integration Tests

This directory contains integration tests that verify PufferLib works correctly with Metta from a fresh installation.

## Overview

The integration tests simulate a new user's experience installing and using Metta with PufferLib. They verify:

1. **Import compatibility** - All necessary imports work without conflicts
2. **Environment creation** - PufferLib environments can be created successfully
3. **Training loops** - Basic training operations complete without errors
4. **Checkpoint handling** - Models can be saved and loaded correctly
5. **Version compatibility** - Both stable and development versions of PufferLib work

## Test Files

### `test_pufferlib_fresh_install.sh`

A comprehensive bash script that creates a fresh virtual environment and tests the entire installation pipeline.

**Usage:**
```bash
# Test with stable PufferLib release
./test_pufferlib_fresh_install.sh

# Test with development branch
./test_pufferlib_fresh_install.sh dev

# Test with specific commit/tag
./test_pufferlib_fresh_install.sh dcd597ef1a094cc2da886f5a4ab2c7f1b27d0183

# Test with specific Python version
./test_pufferlib_fresh_install.sh stable 3.11.7
```

**What it tests:**
- Creates a fresh virtual environment
- Installs PufferLib from specified source
- Builds and installs Metta components
- Verifies all imports work
- Runs a minimal training test
- Tests checkpoint save/load functionality
- Optionally tests `pufferlib train` command

### `test_pufferlib_integration.py`

Pytest-based integration tests for more detailed verification.

**Usage:**
```bash
# Run all integration tests
pytest tests/integration/test_pufferlib_integration.py -v

# Run specific test
pytest tests/integration/test_pufferlib_integration.py::TestPufferLibIntegration::test_vectorized_environment -v

# Run with coverage
pytest tests/integration/test_pufferlib_integration.py --cov=metta
```

**Test coverage:**
- Single and vectorized environment creation
- Environment interaction with policies
- Checkpoint compatibility
- Multi-agent support
- Info dictionary handling
- Multiprocessing vectorization

## CI/CD Integration

The tests are automatically run via GitHub Actions:

- **On Pull Requests** - When changes affect PufferLib integration
- **Daily** - Scheduled runs to catch integration issues early
- **On Demand** - Manual workflow dispatch with custom parameters

See `.github/workflows/pufferlib-integration.yml` for CI configuration.

## Running Tests Locally

### Prerequisites

- Python 3.11.7 (or compatible version)
- `uv` package manager installed
- Git for cloning repositories
- Build tools (gcc, cmake) for compiling C++ extensions

### Quick Start

```bash
# Run the smoke test
cd tests/integration
./test_pufferlib_fresh_install.sh

# Run pytest tests (requires environment setup)
cd ../..  # Back to project root
uv sync
uv run pytest tests/integration/test_pufferlib_integration.py -v
```

### Debugging Failed Tests

If tests fail, check:

1. **Import errors** - Ensure all dependencies are installed correctly
2. **Build errors** - Check C++ compiler and CMake are available
3. **Version conflicts** - Verify compatible versions of dependencies
4. **Environment issues** - Try with a fresh virtual environment

The smoke test script creates temporary directories with detailed logs. Check the output for the test directory path (e.g., `/tmp/metta-pufferlib-test-XXXXX/`).

## Docker Testing

For consistent testing across environments:

```bash
# Build and run tests in Docker
docker build -f .github/workflows/Dockerfile.pufferlib-test -t metta-pufferlib-test .
docker run metta-pufferlib-test
```

## Troubleshooting

### Common Issues

1. **"uv: command not found"**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **C++ build failures**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential cmake
   
   # macOS
   brew install cmake
   ```

3. **PufferLib import errors**
   - Ensure you're using the correct Python version (3.11.7)
   - Try installing from the specific commit in pyproject.toml

### Getting Help

If you encounter issues:

1. Check the CI logs for similar failures
2. Review recent PufferLib changes that might affect integration
3. Open an issue with the full error log and environment details