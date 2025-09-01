from metta.config.schema import get_config
from metta.setup.components.base import SetupModule
from metta.setup.profiles import UserType
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import info


@register_module
class AWSSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "AWS configuration and credentials"

    @property
    def setup_script_location(self) -> str | None:
        if get_saved_settings().user_type.is_softmax:
            return "devops/aws/setup_aws_profiles.sh"
        return None

    def should_install(self) -> bool:
        """Install AWS tools only if user has configured AWS/S3 usage."""
        config = get_config()

        # Check if user needs AWS tools based on their configuration
        needs_aws = (
            config.storage.s3_bucket is not None  # S3 storage configured
            or config.storage.replay_dir
            and "s3://" in config.storage.replay_dir
            or config.storage.torch_profile_dir
            and "s3://" in config.storage.torch_profile_dir
            or config.storage.checkpoint_dir
            and "s3://" in config.storage.checkpoint_dir
            or get_saved_settings().user_type.is_softmax  # Softmax users typically need AWS
        )

        return needs_aws

    def check_installed(self) -> bool:
        try:
            # Also check if AWS CLI is available
            import subprocess

            import boto3  # noqa: F401

            subprocess.run(["aws", "--version"], check=True, capture_output=True)
            return True
        except (ImportError, subprocess.CalledProcessError):
            return False

    def install(self) -> None:
        import platform
        import subprocess

        from metta.setup.utils import success, warning

        saved_settings = get_saved_settings()

        # Install AWS CLI tools first
        if platform.system() == "Darwin":
            try:
                info("Installing AWS CLI tools...")
                subprocess.run(["brew", "tap", "aws/homebrew-aws"], check=True, capture_output=True)
                subprocess.run(["brew", "install", "aws/aws/amazon-efs-utils"], check=True, capture_output=True)
                subprocess.run(["brew", "install", "awscli"], check=True, capture_output=True)
                subprocess.run(["brew", "install", "--cask", "session-manager-plugin"], check=True, capture_output=True)
                success("AWS CLI tools installed")
            except subprocess.CalledProcessError as e:
                warning(f"Failed to install some AWS tools: {e}")

        if saved_settings.user_type == UserType.SOFTMAX_DOCKER:
            info("AWS access for this profile should be provided via IAM roles or environment variables.")
            info("Skipping AWS profile setup.")
            return
        if saved_settings.user_type == UserType.SOFTMAX:
            info("""
                Your AWS access should have been provisioned.
                If you don't have access, contact your team lead.

                Running AWS profile setup...
            """)
            super().install()
        else:
            info("Please configure your AWS credentials using `aws configure` or `aws configure sso`")

    def check_connected_as(self) -> str | None:
        try:
            import boto3

            sts = boto3.client("sts")
            response = sts.get_caller_identity()
            return response["Account"]
        except Exception:
            return None

    def get_configuration_schema(self) -> dict[str, tuple[type, str, str | None]]:
        """Define configuration schema for AWS."""
        return {
            "s3_bucket": (str, "S3 bucket for storing data", None),
            "aws_profile": (str, "AWS profile to use", None),
            "replay_dir": (str, "Directory for replay storage", None),
            "torch_profile_dir": (str, "Directory for torch profiler traces", None),
            "checkpoint_dir": (str, "Directory for model checkpoints", None),
        }

    def interactive_configure(self) -> dict[str, str] | None:
        """Interactive configuration for AWS."""
        from metta.setup.utils import info

        info("\nConfiguring AWS Storage...")
        info("Leave blank to use defaults")

        config = {}

        # Check if connected
        account = self.check_connected_as()
        if account:
            info(f"✓ Connected to AWS account: {account}")

        # Get S3 bucket
        s3_bucket = input("S3 bucket name (e.g., my-company-metta): ").strip()
        if s3_bucket:
            config["s3_bucket"] = s3_bucket

            # Suggest S3 paths if bucket is provided
            use_s3 = input("Use S3 for replays? (y/n) [y]: ").strip().lower()
            if use_s3 != "n":
                config["replay_dir"] = f"s3://{s3_bucket}/replays/"

            use_s3_torch = input("Use S3 for torch profiler traces? (y/n) [n]: ").strip().lower()
            if use_s3_torch == "y":
                config["torch_profile_dir"] = f"s3://{s3_bucket}/torch_traces/"

            use_s3_checkpoints = input("Use S3 for checkpoints? (y/n) [n]: ").strip().lower()
            if use_s3_checkpoints == "y":
                config["checkpoint_dir"] = f"s3://{s3_bucket}/checkpoints/"

        # Get AWS profile
        aws_profile = input("AWS profile name (leave blank for default): ").strip()
        if aws_profile:
            config["aws_profile"] = aws_profile

        return config

    def export_env_vars(self, config: dict[str, str]) -> dict[str, str]:
        """Export AWS configuration as environment variables."""
        env_vars = {}

        if config.get("aws_profile"):
            env_vars["AWS_PROFILE"] = config["aws_profile"]

        if config.get("replay_dir"):
            env_vars["REPLAY_DIR"] = config["replay_dir"]

        if config.get("torch_profile_dir"):
            env_vars["TORCH_PROFILE_DIR"] = config["torch_profile_dir"]

        if config.get("checkpoint_dir"):
            env_vars["CHECKPOINT_DIR"] = config["checkpoint_dir"]

        return env_vars

    def to_config_settings(self) -> dict[str, str | bool]:
        saved_settings = get_saved_settings()

        # Check for configured values first
        from metta.setup.config_manager import get_config_manager

        config = get_config_manager().get_component_config("aws")

        if config.get("replay_dir"):
            return dict(replay_dir=config["replay_dir"])

        # Fall back to profile defaults
        if saved_settings.user_type.is_softmax:
            return dict(replay_dir="s3://softmax-public/replays/")
        return dict(replay_dir="./train_dir/replays/")
