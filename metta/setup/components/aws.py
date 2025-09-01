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

    def check_installed(self) -> bool:
        try:
            import boto3  # noqa: F401

            return True
        except ImportError:
            return False

    def install(self) -> None:
        saved_settings = get_saved_settings()
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
