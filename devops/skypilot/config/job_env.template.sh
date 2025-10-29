#!/usr/bin/env bash
# Environment variables template for SkyPilot jobs
# This file is processed by configure_environment.py

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1

export WANDB_DIR="./wandb"
export WANDB_API_KEY="${WANDB_PASSWORD}"
export DATA_DIR="${DATA_DIR:-./train_dir}"

# Datadog configuration
export DD_ENV="production"
export DD_SERVICE="skypilot-worker"
export DD_AGENT_HOST="localhost"
export DD_TRACE_AGENT_PORT="8126"

export NUM_GPUS="${SKYPILOT_NUM_GPUS_PER_NODE}"
export NUM_NODES="${SKYPILOT_NUM_NODES}"
export MASTER_ADDR="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29501}"
export NODE_INDEX="${SKYPILOT_NODE_RANK}"

# NCCL Configuration
export NCCL_PORT_RANGE="${NCCL_PORT_RANGE:-43000-43063}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

# Debug
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=""

# NCCL Mode
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
