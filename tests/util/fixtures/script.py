#!/usr/bin/env -S uv run
import logging

from metta.common.util.config import Config
from metta.util.metta_script import pydantic_metta_script


class TestScriptConfig(Config):
    run: str


def main(cfg: TestScriptConfig):
    logger = logging.getLogger("test_script")
    logger.info("Hello, world!")
    logger.info(f"Run: {cfg.run}")


pydantic_metta_script(main)
