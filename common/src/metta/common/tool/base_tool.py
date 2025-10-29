"""Base class for SkyPilot tools."""


class BaseTool:
    """Minimal base class for tool implementations."""

    def run(self) -> int:
        """Run the tool. Returns exit code."""
        raise NotImplementedError("Subclasses must implement run()")
