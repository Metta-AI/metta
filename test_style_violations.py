"""Test file with intentional style violations for testing the style review workflow."""

from pathlib import Path


class ConfigManager:
    """Manages configuration files."""

    def __init__(self, config_path: Path):
        """Initialize the config manager.

        Args:
            config_path: Path to the config file

        Returns:
            None
        """
        # Unnecessary defensive check - we know this file exists in our repo
        if config_path.exists():
            self.config_path = config_path
        else:
            raise FileNotFoundError("Config not found")

        # Unnecessary duplication - storing config value separately
        self._name = None

    def load_config(self):
        """Load the configuration from disk."""
        # Inline import - should be at top of file
        from tomllib import load

        with open(self.config_path, "rb") as f:
            return load(f)

    def get_name(self) -> str:
        """Get the configuration name.

        Returns:
            The name from the config
        """
        # Unnecessary indirection
        if self._name is None:
            config = self.load_config()
            self._name = config.get("name", "default")
        return self._name

    def process_data(self, data: list) -> list:
        """Process the data.

        Args:
            data: The input data to process

        Returns:
            list: The processed data
        """
        # Redundant comment that just repeats what the code does
        # Loop through data items
        result = []
        for item in data:
            # Add item to result
            result.append(item.upper())
        return result
