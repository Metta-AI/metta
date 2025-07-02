# Metta API Example

This directory contains an example of using Metta as a library without Hydra configuration.

## Quick Start

1. Install dependencies:
   ```bash
   ./setup_api.sh
   ```

2. Run the example:
   ```bash
   python run.py
   ```

## What it does

The `run.py` script demonstrates:
- Creating environments and agents programmatically
- Full control over the training loop
- Using the same components as the main Metta trainer
- Saving checkpoints and monitoring progress

## Documentation

See `docs/api.md` for complete API documentation.

## Files

- `run.py` - Example training script
- `metta/api.py` - API implementation
- `setup_api.sh` - Setup script to install dependencies
- `docs/api.md` - Full API documentation
