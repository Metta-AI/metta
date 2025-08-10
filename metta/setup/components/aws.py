import json

from metta.setup.components.base import SetupModule
from metta.setup.profiles import UserType
from metta.setup.registry import register_module
from metta.setup.utils import info


@register_module
class AWSSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "AWS configuration and credentials"

    @property
    def setup_script_location(self) -> str | None:
        if self.config.user_type.is_softmax:
            return "devops/aws/setup_aws_profiles.sh"
        return None

    def is_applicable(self) -> bool:
        # Skypilot is a dependency of AWS
        return any(self.config.is_component_enabled(dep) for dep in ["aws", "skypilot"])

    def check_installed(self) -> bool:
        try:
            result = self.run_command(["aws", "--version"], check=False)
        except FileNotFoundError:
            return False
        return result.returncode == 0

    def install(self) -> None:
        if self.config.user_type == UserType.SOFTMAX_DOCKER:
            info("AWS access for this profile should be provided via IAM roles or environment variables.")
            info("Skipping AWS profile setup.")
            return
        if self.config.user_type == UserType.SOFTMAX:
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
            result = self.run_command(["aws", "sts", "get-caller-identity"], check=False)
        except FileNotFoundError:
            return None

        if result.returncode == 0:
            try:
                res = json.loads(result.stdout)
                return res["Account"]
            except Exception:
                pass
        return None
