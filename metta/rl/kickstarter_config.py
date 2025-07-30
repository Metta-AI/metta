from pydantic import Field

from metta.common.util.typed_config import BaseModelWithForbidExtra


class KickstartTeacherConfig(BaseModelWithForbidExtra):
    teacher_uri: str
    # Action loss coefficient: 1.0 gives equal weight to imitating teacher actions
    action_loss_coef: float = Field(default=1, ge=0)
    # Value loss coefficient: 1.0 for standard distillation from teacher values
    value_loss_coef: float = Field(default=1, ge=0)


class KickstartConfig(BaseModelWithForbidExtra):
    teacher_uri: str | None = None
    # Action loss: Weight 1.0 for standard knowledge distillation from teacher
    action_loss_coef: float = Field(default=1, ge=0)
    # Value loss: Weight 1.0 matches action loss for balanced learning
    value_loss_coef: float = Field(default=1, ge=0)
    # Anneal ratio 0.65: Type 2 default chosen arbitrarily
    anneal_ratio: float = Field(default=0.65, ge=0, le=1.0)
    # Kickstart for 1B steps: Type 2 default chosen arbitrarily
    kickstart_steps: int = Field(default=1_000_000_000, gt=0)
    # Additional teachers: Multi-teacher distillation for diverse behaviors
    additional_teachers: list[KickstartTeacherConfig] | None = None
