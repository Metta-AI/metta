from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class KickstartTeacherConfig(BaseModelWithForbidExtra):
    teacher_uri: str
    action_loss_coef: float = Field(default=1, ge=0)
    value_loss_coef: float = Field(default=1, ge=0)


class KickstartConfig(BaseModelWithForbidExtra):
    teacher_uri: str | None = None
    action_loss_coef: float = Field(default=1, ge=0)
    value_loss_coef: float = Field(default=1, ge=0)
    anneal_ratio: float = Field(default=0.65, ge=0, le=1.0)
    kickstart_steps: int = Field(default=1_000_000_000, gt=0)
    additional_teachers: list[KickstartTeacherConfig] | None = None
