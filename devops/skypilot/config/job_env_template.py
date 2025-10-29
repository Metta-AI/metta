"""Environment variable template for SkyPilot jobs.

This module defines the base environment variables that are exported
for all SkyPilot jobs. The configure_environment.py script reads these
and generates shell export statements.
"""

# Core Python environment
PYTHON_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": "${PYTHONPATH:+$PYTHONPATH:}$(pwd)",
    "PYTHONOPTIMIZE": "1",
    "HYDRA_FULL_ERROR": "1",
}

# Weights & Biases configuration
WANDB_ENV = {
    "WANDB_DIR": "./wandb",
    "WANDB_API_KEY": "${WANDB_PASSWORD}",
}

# Data directory
DATA_ENV = {
    "DATA_DIR": "${DATA_DIR:-./train_dir}",
}

# Datadog configuration
DATADOG_ENV = {
    "DD_ENV": "production",
    "DD_SERVICE": "skypilot-worker",
    "DD_AGENT_HOST": "localhost",
    "DD_TRACE_AGENT_PORT": "8126",
}

# Distributed training configuration
DISTRIBUTED_ENV = {
    "NUM_GPUS": "${SKYPILOT_NUM_GPUS_PER_NODE}",
    "NUM_NODES": "${SKYPILOT_NUM_NODES}",
    "MASTER_ADDR": '$(echo "$SKYPILOT_NODE_IPS" | head -n1)',
    "MASTER_PORT": "${MASTER_PORT:-29501}",
    "NODE_INDEX": "${SKYPILOT_NODE_RANK}",
}

# NCCL configuration
NCCL_ENV = {
    "NCCL_PORT_RANGE": "${NCCL_PORT_RANGE:-43000-43063}",
    "NCCL_SOCKET_FAMILY": "${NCCL_SOCKET_FAMILY:-AF_INET}",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}",
    "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "",
    "NCCL_P2P_DISABLE": "${NCCL_P2P_DISABLE:-0}",
    "NCCL_SHM_DISABLE": "${NCCL_SHM_DISABLE:-0}",
    "NCCL_IB_DISABLE": "${NCCL_IB_DISABLE:-1}",
}

# All environment variables in order
ALL_ENV = {
    **PYTHON_ENV,
    **WANDB_ENV,
    **DATA_ENV,
    **DATADOG_ENV,
    **DISTRIBUTED_ENV,
    **NCCL_ENV,
}


def generate_shell_exports() -> str:
    """Generate shell export statements from environment variable definitions."""
    lines = []

    # Add section comments
    sections = [
        ("Python Environment", PYTHON_ENV),
        ("Weights & Biases", WANDB_ENV),
        ("Data Directory", DATA_ENV),
        ("Datadog Configuration", DATADOG_ENV),
        ("Distributed Training", DISTRIBUTED_ENV),
        ("NCCL Configuration", NCCL_ENV),
    ]

    for section_name, env_dict in sections:
        if lines:
            lines.append("")  # Add blank line between sections
        lines.append(f"# {section_name}")
        for key, value in env_dict.items():
            lines.append(f'export {key}="{value}"')

    return "\n".join(lines)
