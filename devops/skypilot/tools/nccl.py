"""NCCL diagnostic tool for SkyPilot.

This tool provides a standardized interface to run NCCL tests on training infrastructure.
Use this to validate GPU communication before starting expensive training runs.
"""

from metta.common.tool.base_tool import BaseTool


class NCCLTool(BaseTool):
    """Tool for running NCCL diagnostic tests."""

    def run(self) -> int:
        """Run NCCL diagnostic tests.

        This executes the standalone NCCL test script which validates:
        - Point-to-point bandwidth between GPU pairs
        - All-reduce collective operation performance
        - GPU topology and NVLink connectivity
        - System diagnostics (NCCL version, CUDA version, driver info)
        """
        import subprocess

        print("=" * 80)
        print("NCCL Diagnostic Tests")
        print("=" * 80)

        # The NCCL test script is standalone and handles its own distributed setup
        # It will use environment variables set by devops/run.sh (NUM_GPUS, etc.)
        result = subprocess.run(
            ["./devops/skypilot/utils/nccl_tests.py"],
            check=False,
        )

        return result.returncode


def main():
    """Entry point for NCCL tool."""
    tool = NCCLTool()
    return tool.run()


if __name__ == "__main__":
    import sys

    sys.exit(main())
