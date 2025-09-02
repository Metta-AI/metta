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

    def install(self, non_interactive: bool = False) -> None:
        """Set up AWS CLI configuration and credentials.

        For softmax-docker profile, skips setup as AWS access should be provided
        via IAM roles or environment variables. For other profiles, provides
        guidance on configuring AWS CLI.

        Args:
            non_interactive: If True, skip interactive configuration prompts
        """
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
            super().install(non_interactive)
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

    def to_config_settings(self) -> dict[str, str | bool]:
        saved_settings = get_saved_settings()
        if saved_settings.user_type.is_softmax:
            return dict(replay_dir="s3://softmax-public/replays/")
        return dict(replay_dir="./train_dir/replays/")
