#!/usr/bin/env -S uv run
import metta.common.config.run_tool as run_tool
from metta.common.util.log_config import init_logging

if __name__ == "__main__":
    init_logging()
    run_tool.main()
