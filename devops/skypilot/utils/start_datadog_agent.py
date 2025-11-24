#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Start Datadog agent with runtime configuration for SkyPilot jobs."""

from metta.setup.components.datadog_agent import DatadogAgentSetup

if __name__ == "__main__":
    setup = DatadogAgentSetup()
    setup.update_log_config_and_start_agent()

