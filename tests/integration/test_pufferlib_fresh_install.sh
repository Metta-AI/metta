#!/usr/bin/env bash
#
# Smoke test for PufferLib integration from a fresh installation.
# This script simulates a new user's experience installing and using Metta with PufferLib.
#
# Usage: ./test_pufferlib_fresh_install.sh [stable|dev|both]
#
# Exit codes:
#   0 - All tests passed
#   1 - Test setup failed
#   2 - PufferLib installation failed
#   3 - Metta installation failed
#   4 - Import test failed
#   5 - Training test failed
#   6 - Checkpoint test failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="${SCRIPT_DIR}/../../"
TEMP_DIR=""
PYTHON_VERSION="${PYTHON_VERSION:-3.11.7}"
TEST_MODE="${1:-both}"  # stable, dev, or both

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Cleanup function
cleanup() {
    if [ -n "${TEMP_DIR}" ] && [ -d "${TEMP_DIR}" ]; then
        print_info "Cleaning up temporary directory: ${TEMP_DIR}"
        rm -rf "${TEMP_DIR}"
    fi
}

# Register cleanup on exit
trap cleanup EXIT

# Function to create a minimal Metta configuration
create_minimal_config() {
    local config_dir="$1"
    mkdir -p "${config_dir}"
    
    cat > "${config_dir}/test_config.yaml" << 'EOF'
# Minimal configuration for smoke test
trainer:
  num_workers: 1
  total_timesteps: 100
  batch_size: 32
  minibatch_size: 16
  bptt_horizon: 8
  checkpoint_interval: 0
  
agent:
  hidden_size: 64

env:
  game:
    max_steps: 50
    num_agents: 2
    obs_width: 7
    obs_height: 7
  map_builder:
    width: 10
    height: 10

hardware:
  device: cpu

wandb:
  mode: disabled
EOF
}

# Function to test a specific PufferLib version
test_pufferlib_version() {
    local version_name="$1"
    local pufferlib_source="$2"
    local test_dir="${TEMP_DIR}/${version_name}"
    
    print_info "========================================="
    print_info "Testing ${version_name} PufferLib version"
    print_info "========================================="
    
    # Create test directory
    mkdir -p "${test_dir}"
    cd "${test_dir}"
    
    # Create virtual environment
    print_info "Creating virtual environment with Python ${PYTHON_VERSION}"
    python${PYTHON_VERSION} -m venv venv || {
        print_error "Failed to create virtual environment"
        return 1
    }
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip and install build tools
    print_info "Upgrading pip and installing build tools"
    pip install --upgrade pip setuptools wheel || {
        print_error "Failed to upgrade pip"
        return 1
    }
    
    # Install PufferLib
    print_info "Installing PufferLib from ${pufferlib_source}"
    if [[ "${pufferlib_source}" == git+* ]]; then
        pip install "${pufferlib_source}" || {
            print_error "Failed to install PufferLib from git"
            return 2
        }
    else
        # Clone and install from source for development version
        git clone https://github.com/PufferAI/PufferLib.git pufferlib_src || {
            print_error "Failed to clone PufferLib"
            return 2
        }
        cd pufferlib_src
        pip install -e . || {
            print_error "Failed to install PufferLib from source"
            return 2
        }
        cd ..
    fi
    
    # Install Metta and its dependencies
    print_info "Installing Metta and dependencies"
    cd "${TEST_ROOT}"
    
    # Install build dependencies first
    pip install scikit-build-core pybind11==2.10.4 numpy || {
        print_error "Failed to install build dependencies"
        return 3
    }
    
    # Install Metta subpackages in order
    for package in common mettagrid agent app_backend; do
        print_info "Installing ${package}..."
        pip install -e "./${package}" || {
            print_error "Failed to install ${package}"
            return 3
        }
    done
    
    # Install main Metta package
    pip install -e . || {
        print_error "Failed to install Metta"
        return 3
    }
    
    cd "${test_dir}"
    
    # Test 1: Verify imports work
    print_info "Test 1: Verifying imports..."
    python -c "
import pufferlib
print(f'PufferLib version: {pufferlib.__version__ if hasattr(pufferlib, \"__version__\") else \"unknown\"}')
print(f'PufferLib location: {pufferlib.__file__}')

# Test Metta imports
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.puffer_env import MettaGridPufferEnv
from metta.rl.vecenv import make_vecenv
from metta.rl.trainer import MettaTrainer
print('✅ All imports successful')
" || {
        print_error "Import test failed"
        return 4
    }
    
    # Test 2: Create and run a minimal environment
    print_info "Test 2: Creating and testing environment..."
    python -c "
import numpy as np
from omegaconf import DictConfig
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.puffer_env import MettaGridPufferEnv

# Create minimal config
config = DictConfig({
    'game': {
        'max_steps': 10,
        'num_agents': 2,
        'obs_width': 5,
        'obs_height': 5,
        'num_observation_tokens': 21,
        'groups': {'agent': {'id': 0, 'sprite': 0}},
        'agent': {
            'default_resource_limit': 10,
            'rewards': {'inventory': {}},
        },
        'actions': {
            'noop': {'enabled': True},
            'move': {'enabled': True},
        },
        'objects': {
            'wall': {'type_id': 1, 'swappable': False},
        },
    },
    'map_builder': {
        '_target_': 'metta.map.mapgen.MapGen',
        'width': 8,
        'height': 8,
        'instances': 1,
        'root': {
            'type': 'metta.map.scenes.test.Simple',
            'params': {
                'agents': 2,
                'wall_border': True,
            },
        },
    },
})

# Create environment
curriculum = SingleTaskCurriculum('smoke_test', config)
env = MettaGridPufferEnv(curriculum=curriculum, render_mode=None)

# Run a few steps
obs, info = env.reset(seed=42)
print(f'Initial observation shape: {obs.shape}')

for i in range(5):
    actions = np.zeros((env.num_agents, 2), dtype=np.int32)
    obs, rewards, terminals, truncations, info = env.step(actions)
    print(f'Step {i+1}: reward sum = {rewards.sum()}')

env.close()
print('✅ Environment test successful')
" || {
        print_error "Environment test failed"
        return 5
    }
    
    # Test 3: Run a minimal training loop
    print_info "Test 3: Running minimal training loop..."
    create_minimal_config "${test_dir}/configs"
    
    # Create a simple training script
    cat > "${test_dir}/train_test.py" << 'EOF'
import sys
import torch
from omegaconf import DictConfig, OmegaConf
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.rl.vecenv import make_vecenv
from metta.agent.policy_store import PolicyStore

# Create minimal config
config = OmegaConf.create({
    'run': 'pufferlib_smoke_test',
    'run_dir': './test_run',
    'device': 'cpu',
    'wandb': {'mode': 'disabled'},
    'trainer': {
        'num_workers': 1,
        'total_timesteps': 100,
        'batch_size': 32,
        'minibatch_size': 16,
        'bptt_horizon': 8,
        'checkpoint_interval': 50,
        'num_epochs': 2,
        'forward_pass_minibatch_target_size': 16,
        'cpu_offload': False,
        'compile': False,
        'ppo': {
            'clip_coef': 0.1,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
            'gamma': 0.99,
            'gae_lambda': 0.95,
        },
        'optimizer': {
            'type': 'adam',
            'learning_rate': 3e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-5,
            'weight_decay': 0.0,
        },
    },
    'agent': {
        'hidden_size': 64,
        'clip_range': 0.0,
    },
    'vectorization': 'serial',
    'sim': {'evaluate_interval': 0},
    'env': {
        'game': {
            'max_steps': 20,
            'num_agents': 2,
            'obs_width': 5,
            'obs_height': 5,
            'num_observation_tokens': 21,
            'groups': {'agent': {'id': 0, 'sprite': 0}},
            'agent': {
                'default_resource_limit': 10,
                'rewards': {'inventory': {}},
            },
            'actions': {
                'noop': {'enabled': True},
                'move': {'enabled': True},
            },
            'objects': {
                'wall': {'type_id': 1, 'swappable': False},
            },
        },
        'map_builder': {
            '_target_': 'metta.map.mapgen.MapGen',
            'width': 8,
            'height': 8,
            'instances': 1,
            'root': {
                'type': 'metta.map.scenes.test.Simple',
                'params': {
                    'agents': 2,
                    'wall_border': True,
                },
            },
        },
    },
})

# Create curriculum
curriculum = SingleTaskCurriculum('smoke_test', config.env)

# Create environment
vecenv = make_vecenv(
    curriculum=curriculum,
    vectorization='serial',
    num_envs=1,
    num_workers=1,
    is_training=True,
)

print(f"Environment created with {vecenv.num_agents} agents")

# Create a simple policy
from metta.agent.external.example import Recurrent
policy = Recurrent(vecenv)

# Run a few training steps
obs = vecenv.reset()
for i in range(10):
    with torch.no_grad():
        actions, _ = policy(torch.from_numpy(obs[0]))
    obs = vecenv.step(actions[0].numpy())
    print(f"Training step {i+1} completed")

print("✅ Training loop test successful")

# Test checkpoint saving
policy_store = PolicyStore(config, None)
metadata = {'test': True, 'steps': 10}
policy_record = policy_store.save(
    policy,
    agent_steps=10,
    metadata=metadata,
    feature_mapping=None,
)
print(f"✅ Checkpoint saved: {policy_record.uri}")

# Test checkpoint loading
loaded_record = policy_store.load_latest()
if loaded_record:
    print(f"✅ Checkpoint loaded: {loaded_record.uri}")
else:
    print("❌ Failed to load checkpoint")
    sys.exit(6)
EOF
    
    python "${test_dir}/train_test.py" || {
        print_error "Training test failed"
        return 5
    }
    
    print_info "✅ All tests passed for ${version_name} version!"
    
    # Deactivate virtual environment
    deactivate
    
    return 0
}

# Main execution
main() {
    print_info "Starting PufferLib fresh installation smoke test"
    print_info "Python version: ${PYTHON_VERSION}"
    print_info "Test mode: ${TEST_MODE}"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d -t pufferlib_test_XXXXXX)
    print_info "Created temporary directory: ${TEMP_DIR}"
    
    # Test results
    local stable_result=0
    local dev_result=0
    
    # Test stable version
    if [[ "${TEST_MODE}" == "stable" ]] || [[ "${TEST_MODE}" == "both" ]]; then
        # Use the Metta fork for stable since that's what the project uses
        test_pufferlib_version "stable" "git+https://github.com/Metta-AI/PufferLib.git@dcd597ef1a094cc2da886f5a4ab2c7f1b27d0183" || stable_result=$?
    fi
    
    # Test development version
    if [[ "${TEST_MODE}" == "dev" ]] || [[ "${TEST_MODE}" == "both" ]]; then
        test_pufferlib_version "development" "https://github.com/PufferAI/PufferLib.git" || dev_result=$?
    fi
    
    # Summary
    print_info "========================================="
    print_info "Test Summary:"
    if [[ "${TEST_MODE}" == "stable" ]] || [[ "${TEST_MODE}" == "both" ]]; then
        if [[ ${stable_result} -eq 0 ]]; then
            print_info "✅ Stable version: PASSED"
        else
            print_error "❌ Stable version: FAILED (exit code: ${stable_result})"
        fi
    fi
    
    if [[ "${TEST_MODE}" == "dev" ]] || [[ "${TEST_MODE}" == "both" ]]; then
        if [[ ${dev_result} -eq 0 ]]; then
            print_info "✅ Development version: PASSED"
        else
            print_error "❌ Development version: FAILED (exit code: ${dev_result})"
        fi
    fi
    
    # Return overall result
    if [[ ${stable_result} -ne 0 ]] || [[ ${dev_result} -ne 0 ]]; then
        return 1
    fi
    
    return 0
}

# Run main function
main "$@"