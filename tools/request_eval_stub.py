#!/usr/bin/env -S uv run
"""
Temporary stub for request_eval.py - evaluation request system needs full refactor.

The original request_eval.py used complex PolicyStore functionality for:
- Multi-policy discovery across wandb runs
- Policy selection strategies (top, latest, random)
- External stats server integration for ranking
- Complex parallel policy loading

This stub provides basic functionality until a full migration is completed.
TODO: Implement full functionality with CheckpointManager-based discovery
"""

import logging
import sys

logger = logging.getLogger(__name__)


def main():
    """Temporary main function for evaluation requests."""
    logger.warning("request_eval.py has been temporarily disabled due to PolicyStore migration")
    logger.info("Use the tools/sim.py script for basic policy evaluation")
    logger.info("Full request_eval functionality will be restored in a future update")
    return 1


if __name__ == "__main__":
    sys.exit(main())
