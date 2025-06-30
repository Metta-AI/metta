# Import all component modules to ensure they register themselves
from devops.setup.components.aws import AWSSetup
from devops.setup.components.core import CoreSetup
from devops.setup.components.githooks import GitHooksSetup
from devops.setup.components.mettascope import MettaScopeSetup
from devops.setup.components.observatory_cli import ObservatoryCliSetup
from devops.setup.components.observatory_fe import ObservatoryFeSetup
from devops.setup.components.skypilot import SkypilotSetup
from devops.setup.components.system import SystemSetup
from devops.setup.components.tailscale import TailscaleSetup
from devops.setup.components.wandb import WandbSetup

__all__ = [
    "AWSSetup",
    "CoreSetup",
    "GitHooksSetup",
    "MettaScopeSetup",
    "ObservatoryCliSetup",
    "ObservatoryFeSetup",
    "SkypilotSetup",
    "SystemSetup",
    "TailscaleSetup",
    "WandbSetup",
]
