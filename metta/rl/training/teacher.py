from __future__ import annotations

from typing import Literal

from pydantic import Field

from mettagrid.base_config import Config


class TeacherConfig(Config):
    """Shared knobs for enabling teacher/supervisor driven training phases."""

    policy_uri: str | None = None
    mode: Literal["sliced_cloner", "supervisor"] = "sliced_cloner"
    steps: int | None = None
    teacher_led_proportion: float = Field(default=1.0, ge=0.0, le=1.0)
    student_led_proportion: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def enabled(self) -> bool:
        return self.policy_uri is not None
