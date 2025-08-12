#!/usr/bin/env python3

from typing import Any, Union

from pydantic import BaseModel, Field


class TestConfig(BaseModel):
    env: Union[str, Any, None] = Field(default=None)


# Test
config_dict = {"env": "/env/test"}

print("Testing simple Union validation...")
try:
    config = TestConfig.model_validate(config_dict)
    print(f"SUCCESS: env = {config.env}, type = {type(config.env)}")
except Exception as e:
    print(f"ERROR: {e}")
