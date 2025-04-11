import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import requests
import json

def config_from_path(config_path: str, overrides: DictConfig = None) -> DictConfig:
    env_cfg = hydra.compose(config_name=config_path)
    if config_path.startswith("/"):
        config_path = config_path[1:]
    path = config_path.split("/")
    for p in path[:-1]:
        env_cfg = env_cfg[p]
    if overrides is not None:
        env_cfg = OmegaConf.merge(env_cfg, overrides)
    return env_cfg

def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def check_ec2() -> bool:
    """Check if we're running on an EC2 instance with valid IAM role credentials."""
    try:
        # First try IMDSv2 (more secure)
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=1
        )
        if token_response.status_code == 200:
            token = token_response.text
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=1
            )
            if response.status_code == 200 and response.text:
                return True

        # Fallback to IMDSv1 if IMDSv2 fails
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            timeout=1
        )
        return response.status_code == 200 and response.text
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return False

def setup_metta_environment(
    cfg: DictConfig,
    require_aws: bool = True,
    require_wandb: bool = True
):
    if require_aws:
        # Check that AWS is good to go.
        # Check that ~/.aws/credentials exist or env var AWS_PROFILE is set.
        if not os.path.exists(os.path.expanduser("~/.aws/sso/cache")) and \
            "AWS_ACCESS_KEY_ID" not in os.environ and \
            "AWS_SECRET_ACCESS_KEY" not in os.environ and \
            not check_ec2():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("python ./devops/aws/setup_sso.py")
            print("Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            exit(1)
    if cfg.wandb.track and require_wandb:
        # Check that W&B is good to go.
        # Open ~/.netrc file and see if there is a api.wandb.ai entry.
        if "api.wandb.ai" not in read_file(os.path.expanduser("~/.netrc")) and \
            "WANDB_API_KEY" not in os.environ:
            print("W&B is not configured, please install:")
            print("pip install wandb")
            print("and run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
