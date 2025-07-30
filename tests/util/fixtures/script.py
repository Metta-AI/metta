#!/usr/bin/env -S uv run
import logging

from omegaconf import DictConfig

from metta.util.metta_script import metta_script


def main(cfg: DictConfig):
    logger = logging.getLogger("test_script")
    logger.info("Hello, world!")


metta_script(main, "replay_job")
