from __future__ import annotations

from omegaconf import DictConfig, ListConfig

from metta.util.aws import check_aws_credentials
from metta.util.wandb.credentials import check_wandb_credentials


def setup_metta_environment(cfg: ListConfig | DictConfig, require_aws: bool = True, require_wandb: bool = True):
    if require_aws:
        # Check that AWS is good to go.
        if not check_aws_credentials():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("aws sso login --profile softmax")
            print("Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            exit(1)
    if cfg.wandb.enabled and require_wandb:
        # Check that W&B is good to go.
        if not check_wandb_credentials():
            print("W&B is not configured, please run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
