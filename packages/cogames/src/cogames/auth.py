"""Shared authentication functionality for CLI login scripts."""

import asyncio
import os
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


class BaseCLIAuthenticator:
    """Base class for CLI authentication with OAuth2 flow.

    This class handles the OAuth2 authentication flow for CLI tools,
    including starting a local callback server and saving tokens.
    """

    def __init__(
        self,
        auth_server_url: str,
        token_file_name: str,
        token_storage_key: str | None = None,
        extra_uris: dict[str, list[str]] | None = None,
    ):
        """Initialize the authenticator.

        Args:
            auth_server_url: Base URL of the authentication server
            token_file_name: Name of the YAML file to store tokens (e.g., 'observatory_tokens.yaml')
            token_storage_key: Optional key to nest tokens under in YAML (e.g., 'login_tokens').
                             If None, tokens are stored at the top level.
            extra_uris: Optional dict mapping auth server URLs to lists of additional URIs
                       that should receive the same token
        """
        self.auth_url = auth_server_url + "/tokens/cli"
        self.auth_server_url = auth_server_url
        self.token_storage_key = token_storage_key
        self.extra_uris = extra_uris or {}
        self.token = None
        self.error = None
        self.server_started = threading.Event()
        self.auth_completed = threading.Event()

        home = Path.home()
        self.config_dir = home / ".metta"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_file = self.config_dir / token_file_name

    def create_app(self) -> FastAPI:
        """Create the FastAPI application for handling OAuth2 callbacks."""
        app = FastAPI(title="CLI OAuth2 Callback Server")

        @app.get("/callback")
        async def callback(request: Request):
            """Handle the OAuth2 callback"""
            try:
                # Get token from query parameters
                token = request.query_params.get("token")

                if not token:
                    self.error = "No token received in callback"
                    self.auth_completed.set()
                    return HTMLResponse(content=self._error_html("No token received"), status_code=400)

                # Store the token
                self.token = token
                self.auth_completed.set()

                return HTMLResponse(content=self._success_html())

            except Exception as e:
                self.error = f"Callback error: {str(e)}"
                self.auth_completed.set()
                return HTMLResponse(content=self._error_html(f"Error: {str(e)}"), status_code=500)

        return app

    def _success_html(self) -> str:
        """HTML response for successful authentication"""
        return """
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                .container { max-width: 500px; margin: 0 auto; padding: 20px; }
                .success { color: #28a745; }
                .close-btn {
                    background: #007bff; color: white; border: none;
                    padding: 10px 20px; border-radius: 5px; cursor: pointer;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="success">Authentication Successful!</h1>
                <p>You can close this window and return to the CLI.</p>
                <button class="close-btn" onclick="window.close()">Close Window</button>
            </div>
            <script>
                // Auto-close after 3 seconds
                setTimeout(() => window.close(), 3000);
            </script>
        </body>
        </html>
        """

    def _error_html(self, error_message: str) -> str:
        """HTML response for authentication errors"""
        return f"""
        <html>
        <head>
            <title>Authentication Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }}
                .container {{ max-width: 500px; margin: 0 auto; padding: 20px; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="error">Authentication Failed</h1>
                <p>{error_message}</p>
                <p>Please try again or contact support.</p>
            </div>
        </body>
        </html>
        """

    def _find_free_port(self) -> int:
        """Find a free port to bind the server to"""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _build_auth_url(self, callback_url: str) -> str:
        """Build the authentication URL with callback parameter"""
        params = {
            "callback": callback_url,
        }

        return f"{self.auth_url}?{urlencode(params)}"

    def _open_browser(self, url: str) -> None:
        """Open the default browser to the authentication URL"""
        if not webbrowser.open(url):
            print("Failed to open browser automatically")
            print(f"Please manually visit: {url}")

    def save_token(self, token: str) -> None:
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
            token_data = {self.auth_server_url: token}
            if extra_uris := self.extra_uris.get(self.auth_server_url):
                for uri in extra_uris:
                    token_data[uri] = token

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

            print(f"Token saved for {self.auth_server_url}")

        except Exception as e:
            raise Exception(f"Failed to save token: {e}") from e

    def _run_server(self, port: int) -> None:
        """Run the FastAPI server in a separate thread"""
        try:
            app = self.create_app()
            config = uvicorn.Config(
                app=app,
                host="127.0.0.1",
                port=port,
                log_level="error",  # Suppress uvicorn logs
                access_log=False,
            )
            server = uvicorn.Server(config)

            # Signal that server is ready
            self.server_started.set()

            # Run server (this blocks until shutdown)
            asyncio.run(server.serve())

        except Exception as e:
            self.error = f"Server error: {e}"
            self.server_started.set()
            self.auth_completed.set()

    def authenticate(self, timeout: int = 300) -> bool:
        """Perform the OAuth2 authentication flow.

        Args:
            timeout: Maximum time to wait for authentication (seconds)

        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Find a free port
            port = self._find_free_port()
            callback_url = f"http://127.0.0.1:{port}/callback"

            print(f"Starting local callback server on port {port}")

            # Start server in background thread
            server_thread = threading.Thread(target=self._run_server, args=(port,), daemon=True)
            server_thread.start()

            # Wait for server to start
            if not self.server_started.wait(timeout=10):
                raise Exception("Server failed to start within 10 seconds")

            if self.error:
                raise Exception(self.error)

            # Build auth URL and open browser
            auth_url = self._build_auth_url(callback_url)
            print(f"Opening browser to: {auth_url}")
            self._open_browser(auth_url)

            # Wait for authentication to complete
            print("Waiting for authentication...")
            if not self.auth_completed.wait(timeout=timeout):
                raise Exception(f"Authentication timed out after {timeout} seconds")

            # Check for errors
            if self.error:
                raise Exception(self.error)

            if not self.token:
                raise Exception("No token received")

            # Save token
            self.save_token(self.token)

            return True

        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def has_saved_token(self) -> bool:
        """Check if we have a saved token for this server"""
        if self.yaml_file.exists():
            try:
                with open(self.yaml_file, "r") as f:
                    data = yaml.safe_load(f) or {}

                # Get the token dictionary based on storage structure
                if self.token_storage_key:
                    tokens = data.get(self.token_storage_key, {})
                else:
                    tokens = data

                # Check all required URLs have tokens
                all_urls = [self.auth_server_url] + self.extra_uris.get(self.auth_server_url, [])
                return all(url in tokens for url in all_urls)
            except Exception:
                pass

        return False
