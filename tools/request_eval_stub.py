#!/usr/bin/env -S uv run
"""Evaluation request system stub."""

import logging
import sys

logger = logging.getLogger(__name__)


def main():
    """Main function for evaluation requests."""
    logger.warning("request_eval.py functionality not yet implemented")
    logger.info("Use tools/sim.py for policy evaluation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
