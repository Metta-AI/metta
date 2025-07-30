from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class BaseModelWithForbidExtra(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
