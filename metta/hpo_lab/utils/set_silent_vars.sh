#!/bin/bash
# HPO Lab Environment Setup - Silences common warnings
# Source this file to set environment variables for the current session:
#   source ./metta/hpo_lab/utils/set_silent_vars.sh

echo "Setting environment variables to suppress warnings..."

# Suppress PyTorch distributed warnings on macOS/Windows
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Suppress Python warnings (UserWarning, DeprecationWarning)
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# Suppress Pydantic field attribute warnings
export PYDANTIC_SILENCE_WARNINGS=1

# Suppress pygame pkg_resources deprecation warning
export PYGAME_HIDE_SUPPORT_PROMPT=1

# Suppress TensorFlow/JAX warnings if they appear
export TF_CPP_MIN_LOG_LEVEL=2
export JAX_PLATFORM_NAME=cpu

# Suppress wandb warnings
export WANDB_SILENT=true
export WANDB_DISABLE_CODE=true

# Suppress Ray warnings
export RAY_DISABLE_PYARROW_VERSION_CHECK=1
export RAY_WARNINGS_DISABLED=1

# Suppress gymnasium warnings
export GYMNASIUM_DISABLE_ENV_CHECKER=1

echo "✅ Environment configured for silent operation"
echo ""
echo "Active suppressions:"
echo "  • PyTorch distributed warnings"
echo "  • Pydantic field attribute warnings"
echo "  • Python UserWarnings and DeprecationWarnings"
echo "  • Pygame pkg_resources warnings"
echo "  • Ray and WandB verbose output"
echo ""
echo "To make this permanent, add this line to your ~/.bashrc or ~/.zshrc:"
echo "  source $(pwd)/metta/hpo_lab/utils/set_silent_vars.sh"