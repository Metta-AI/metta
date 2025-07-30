from enum import Enum

from typing_extensions import NotRequired, TypedDict

from metta.common.util.constants import METTA_AWS_ACCOUNT_ID, METTA_SKYPILOT_URL, METTA_WANDB_ENTITY


class UserType(Enum):
    EXTERNAL = "external"
    CLOUD = "cloud"
    SOFTMAX = "softmax"
    SOFTMAX_DOCKER = "softmax-docker"
    CUSTOM = "custom"

    @property
    def is_softmax(self) -> bool:
        return self in (UserType.SOFTMAX, UserType.SOFTMAX_DOCKER)

    def get_description(self) -> str:
        descriptions = {
            UserType.EXTERNAL: "External contributor",
            UserType.CLOUD: "User with own cloud account",
            UserType.SOFTMAX: "Softmax employee",
            UserType.SOFTMAX_DOCKER: "Softmax (Docker)",
            UserType.CUSTOM: "Custom configuration",
        }
        return descriptions.get(self, self.value)


class ComponentConfig(TypedDict):
    enabled: bool
    expected_connection: NotRequired[str | None]


class ProfileConfig(TypedDict):
    components: dict[str, ComponentConfig]


PROFILE_DEFINITIONS: dict[UserType, ProfileConfig] = {
    UserType.EXTERNAL: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "nodejs": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": False},
            "wandb": {"enabled": False},
            "skypilot": {"enabled": False},
            "tailscale": {"enabled": False},
            "heatmapwidget": {"enabled": True},
        }
    },
    UserType.CLOUD: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "nodejs": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": True},
            "wandb": {"enabled": True},
            "skypilot": {"enabled": True},
            "tailscale": {"enabled": False},
            "heatmapwidget": {"enabled": False},
        }
    },
    UserType.SOFTMAX_DOCKER: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "nodejs": {"enabled": False},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": False},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": True, "expected_connection": METTA_AWS_ACCOUNT_ID},
            "wandb": {"enabled": True, "expected_connection": METTA_WANDB_ENTITY},
            "skypilot": {"enabled": False},
            "tailscale": {"enabled": False},
            "heatmapwidget": {"enabled": False},
        }
    },
    UserType.SOFTMAX: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "codeclip": {"enabled": True},
            "nodejs": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": True, "expected_connection": "@stem.ai"},
            "aws": {"enabled": True, "expected_connection": METTA_AWS_ACCOUNT_ID},
            "wandb": {"enabled": True, "expected_connection": METTA_WANDB_ENTITY},
            "skypilot": {"enabled": True, "expected_connection": METTA_SKYPILOT_URL},
            "tailscale": {"enabled": True, "expected_connection": "@stem.ai"},
            "heatmapwidget": {"enabled": True},
        }
    },
}
