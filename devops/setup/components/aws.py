import json
import subprocess
from pathlib import Path

from devops.setup.components.base import SetupModule
from devops.setup.config import UserType
from devops.setup.registry import register_module
from devops.setup.utils import info


@register_module
class AWSSetup(SetupModule):
    @property
    def description(self) -> str:
        return "AWS configuration and credentials"

    @property
    def setup_script_location(self) -> str | None:
        if self.config.user_type == UserType.SOFTMAX:
            return "devops/aws/setup_aws_profiles.sh"
        return None

    def is_applicable(self) -> bool:
        return self.config.user_type in [
            UserType.SOFTMAX,
            UserType.CLOUD,
        ] and self.config.is_component_enabled("aws")

    def check_installed(self) -> bool:
        aws_config = Path.home() / ".aws" / "config"
        if not aws_config.exists():
            return False

        return True

    def install(self) -> None:
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
        if not self.check_installed():
            return None
        result = subprocess.run(["aws", "sts", "get-caller-identity"], capture_output=True, text=True)

        if result.returncode == 0:
            try:
                res = json.loads(result.stdout)
                return res["Account"]
            except Exception:
                pass
        return None
