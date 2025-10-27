"""Shared authentication functionality for CLI login scripts."""

import asyncio
import html
import os
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence
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

    def _render_html(
        self,
        *,
        title: str,
        headline: str,
        message_lines: Sequence[str],
        status: Literal["success", "error"],
        auto_close_seconds: int | None = None,
        extra_html: str = "",
    ) -> str:
        """Render a styled HTML page for the OAuth callback."""
        icon = "&#10003;" if status == "success" else "&#9888;"
        escaped_title = html.escape(title)
        escaped_headline = html.escape(headline)
        messages = "".join(f"<p class='smx-auth__message'>{html.escape(line)}</p>" for line in message_lines)
        current_year = datetime.now().year
        auto_close_script = ""
        if auto_close_seconds is not None:
            auto_close_script = f"""
        <script>
            window.setTimeout(function () {{
                try {{
                    window.close();
                }} catch (err) {{
                    console.debug("Auto-close suppressed", err);
                }}
            }}, {int(auto_close_seconds * 1000)});
        </script>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escaped_title}</title>
    <link rel="stylesheet" href="https://softmax.com/Assets/softmax.css" />
    <style>
        :root {{
            color-scheme: light;
        }}
        body.smx-auth-page {{
            margin: 0;
            min-height: 100vh;
            width: 100%;
            background-color: #fffdf4;
            color: #0E2758;
            font-family: "ABC Marfa Variable", "Roboto", -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: clamp(2.4rem, 8vw, 4rem) 1.5rem;
            text-rendering: optimizeLegibility;
        }}
        .smx-auth-card {{
            width: min(560px, 100%);
            background: rgba(255, 254, 248, 0.95);
            border-radius: 24px;
            border: 1px solid rgba(14, 39, 88, 0.12);
            box-shadow: 0 32px 60px rgba(14, 39, 88, 0.12);
            padding: clamp(2rem, 6vw, 3.25rem);
            text-align: center;
        }}
        .smx-auth-card--success .smx-auth-icon {{
            background: rgba(26, 107, 63, 0.16);
            color: #195C38;
        }}
        .smx-auth-card--error .smx-auth-icon {{
            background: rgba(176, 46, 38, 0.16);
            color: #952F2B;
        }}
        .smx-auth-icon {{
            height: 76px;
            width: 76px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin: 0 auto 20px;
            border: 1px solid rgba(14, 39, 88, 0.12);
        }}
        .smx-auth-headline {{
            margin: 0 0 12px;
            font-size: clamp(1.8rem, 5vw, 2.35rem);
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        .smx-auth__message {{
            margin: 0 0 12px;
            font-size: 1.02rem;
            line-height: 1.6;
            color: rgba(14, 39, 88, 0.72);
        }}
        .smx-auth__body {{
            display: grid;
            gap: 8px;
        }}
        .smx-auth__actions {{
            margin-top: 32px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .smx-auth-button {{
            appearance: none;
            border-radius: 999px;
            border: 2px solid #0E2758;
            background: #0E2758;
            color: #fffdf4;
            cursor: pointer;
            padding: 0.9rem 1.8rem;
            font-size: 0.95rem;
            font-family: "Marfa Mono", "Courier New", monospace;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            transition: transform 0.15s ease, box-shadow 0.2s ease, background 0.2s ease;
        }}
        .smx-auth-button:hover {{
            transform: translateY(-1px);
            background: #1a3875;
            border-color: #1a3875;
            box-shadow: 0 14px 28px rgba(26, 56, 117, 0.18);
        }}
        .smx-auth-button:active {{
            transform: translateY(0);
            box-shadow: 0 8px 16px rgba(14, 39, 88, 0.18);
        }}
        .smx-auth-footnote {{
            margin-top: 28px;
            font-size: 0.85rem;
            color: rgba(14, 39, 88, 0.55);
        }}
        @media (max-width: 540px) {{
            .smx-auth-card {{
                padding: 2.4rem 1.8rem;
            }}
        }}
    </style>
</head>
<body class="smx-auth-page">
    <main class="smx-auth-card smx-auth-card--{status}" role="dialog" aria-live="polite">
        <div class="smx-auth-icon" aria-hidden="true">{icon}</div>
        <h1 class="smx-auth-headline">{escaped_headline}</h1>
        <div class="smx-auth__body">
            {messages}
            {extra_html}
        </div>
        <div class="smx-auth-footnote">may we all find alignment - softmax, {current_year}</div>
    </main>
    {auto_close_script}
</body>
</html>"""

    def _success_html(self) -> str:
        """HTML response for successful authentication"""
        return self._render_html(
            title="Authentication Successful",
            headline="You're all set!",
            message_lines=[
                "Authentication complete. You can return to the terminal.",
                "This window will close automatically in a moment.",
            ],
            status="success",
            auto_close_seconds=3,
            extra_html="""
            <div class="smx-auth__actions">
                <button class="smx-auth-button" type="button" onclick="window.close()">Close this window</button>
            </div>
            """,
        )

    def _error_html(self, error_message: str) -> str:
        """HTML response for authentication errors"""
        return self._render_html(
            title="Authentication Error",
            headline="Something went wrong",
            message_lines=[
                error_message,
                "Please retry the login process or contact support if the issue persists.",
            ],
            status="error",
        )

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

    def load_token(self) -> str | None:
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

            return tokens.get(self.auth_server_url)
        except Exception:
            return None

    def has_saved_token(self) -> bool:
        """Check if we have a saved token for this server"""
        token = self.load_token()
        if token is None:
            return False
        return True
