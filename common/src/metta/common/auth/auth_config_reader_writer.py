import os
from pathlib import Path

import yaml


class AuthConfigReaderWriter:
    def __init__(self, token_file_name: str, token_storage_key: str | None = None):
        home = Path.home()
        self.config_dir = home / ".metta"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_file = self.config_dir / token_file_name

        self.token_file_name = token_file_name
        self.token_storage_key = token_storage_key

    def load_token(self, token_key: str) -> str | None:
        """Load the token for this auth server from the YAML file.

        Returns the token string if found, None otherwise.
        """
        if not self.yaml_file.exists():
            return None

        try:
            with open(self.yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}

            # Get the token dictionary based on storage structure
            if self.token_storage_key:
                tokens = data.get(self.token_storage_key, {})
            else:
                tokens = data

            return tokens.get(token_key)
        except Exception:
            return None

    def has_saved_token(self, token_key: str) -> bool:
        """Check if we have a saved token for this server"""
        token = self.load_token(token_key)
        if token is None:
            return False
        return True

    def save_token(self, token: str, auth_server_key: str) -> None:
        """Save the token to a YAML file with secure permissions.

        If token_storage_key is set, tokens are nested under that key.
        Otherwise, they are stored at the top level.
        """
        try:
            # Read existing data
            existing_data = {}
            if self.yaml_file.exists():
                with open(self.yaml_file, "r") as f:
                    existing_data = yaml.safe_load(f) or {}

            # Prepare token data
            token_data = {auth_server_key: token}

            # Update data structure based on token_storage_key
            if self.token_storage_key:
                # Nested structure: {token_storage_key: {url: token}}
                if self.token_storage_key not in existing_data:
                    existing_data[self.token_storage_key] = {}
                existing_data[self.token_storage_key].update(token_data)
            else:
                # Flat structure: {url: token}
                existing_data.update(token_data)

            # Write all data back
            with open(self.yaml_file, "w") as f:
                yaml.safe_dump(existing_data, f, default_flow_style=False)

            # Set secure permissions (readable only by owner)
            os.chmod(self.yaml_file, 0o600)

            print(f"Token saved for {auth_server_key}")

        except Exception as e:
            raise Exception(f"Failed to save token: {e}") from e


observatory_auth_config = AuthConfigReaderWriter(token_file_name="config.yaml", token_storage_key="observatory_tokens")
