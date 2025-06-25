from pydantic import BaseModel


class BaseModelWithForbidExtra(BaseModel):
    model_config = dict(extra="forbid")  # pyright: ignore[reportAssignmentType]
