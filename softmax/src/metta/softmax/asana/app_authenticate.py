#!/usr/bin/env -S uv run

import asyncio
import os
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from metta.softmax.aws.secrets_manager import get_secret

ASANA_AUTH_URL = "https://app.asana.com/-/oauth_authorize"
ASANA_TOKEN_URL = "https://app.asana.com/-/oauth_token"


class _LocalAsanaTokens:
    config_dir: Path = Path.home() / ".metta"
    yaml_file: Path = config_dir / "asana_tokens.yaml"

    @classmethod
    def load_token(cls, client_id: str) -> dict[str, Any] | None:
        return cls.load_tokens().get(client_id)

    @classmethod
    def load_tokens(cls) -> dict[str, Any]:
        cls.config_dir.mkdir(parents=True, exist_ok=True)
        if cls.yaml_file.exists():
            with open(cls.yaml_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @classmethod
    def save_token(cls, client_id: str, token: dict[str, Any]) -> None:
        """Save tokens to YAML file with secure permissions"""
        try:
            now = int(time.time())
            expires_in = int(token.get("expires_in", 3600))

            record = {
                "client_id": client_id,
                "access_token": token.get("access_token"),
                "refresh_token": token.get("refresh_token"),
                "token_type": token.get("token_type", "Bearer"),
                "expires_in": expires_in,
                "expires_at": now + max(0, expires_in - 60),
                "raw": token,
                "saved_at": now,
            }

            # Update with new token
            existing = cls.load_tokens()
            existing[client_id] = record

            # Write all tokens back
            with open(cls.yaml_file, "w") as f:
                yaml.safe_dump(existing, f, default_flow_style=False)

            # Set secure permissions (readable only by owner)
            os.chmod(cls.yaml_file, 0o600)

            print(f"Tokens saved to {cls.yaml_file}")

        except Exception as e:
            raise Exception(f"Failed to save tokens: {e}") from e


class AsanaOAuthCLI:
    config_dir: Path = Path.home() / ".metta"
    yaml_file: Path = config_dir / "asana_tokens.yaml"

    def __init__(self, name: str):
        credentials = get_secret(f"asana/{name}_app")
        self.client_id = credentials["client_id"]
        self.client_secret = credentials["client_secret"]
        self.state = None
        self.code = None
        self.error = None
        self.server_started = threading.Event()
        self.auth_completed = threading.Event()
        # Runtime handles for graceful shutdown
        self._server = None
        self._server_thread = None

    def _request_shutdown(self) -> None:
        """Signal the uvicorn server to shut down if it's running."""
        try:
            if self._server is not None:
                self._server.should_exit = True  # type: ignore[attr-defined]
        except Exception:
            pass

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Asana OAuth Callback Server")

        @app.get("/callback")
        async def callback(request: Request):
            """Handle the OAuth2 callback"""
            try:
                params = request.query_params

                # Check for error from Asana
                if error := params.get("error"):
                    self.error = f"Error from Asana: {error}"
                    self.auth_completed.set()
                    self._request_shutdown()
                    return HTMLResponse(content=self._error_html(self.error), status_code=400)

                # Get code and state
                code = params.get("code")
                state = params.get("state")

                if not code:
                    self.error = "No authorization code received"
                    self.auth_completed.set()
                    self._request_shutdown()
                    return HTMLResponse(content=self._error_html(self.error), status_code=400)

                # Verify state for CSRF protection
                if not secrets.compare_digest(state or "", self.state or ""):
                    self.error = "Invalid state parameter (CSRF protection)"
                    self.auth_completed.set()
                    self._request_shutdown()
                    return HTMLResponse(content=self._error_html(self.error), status_code=400)

                # Store the code
                self.code = code
                self.auth_completed.set()
                self._request_shutdown()

                return HTMLResponse(content=self._success_html())

            except Exception as e:
                self.error = f"Callback error: {str(e)}"
                self.auth_completed.set()
                self._request_shutdown()
                return HTMLResponse(content=self._error_html(str(e)), status_code=500)

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
                <h1 class="success">✓ Authentication Successful!</h1>
                <p>You can close this window and return to the CLI.</p>
                <button class="close-btn" onclick="window.close()">Close Window</button>
            </div>
            <script>setTimeout(() => window.close(), 3000);</script>
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
                <h1 class="error">✗ Authentication Failed</h1>
                <p>{error_message}</p>
                <p>Please try again or check your Asana app settings.</p>
            </div>
        </body>
        </html>
        """

    def _build_auth_url(self, redirect_uri: str, state: str) -> str:
        """Build the Asana authorization URL"""
        # Define scopes your app needs
        scopes = [
            "tasks:read",  # Basic API access
            "projects:read",  # Basic API access
            "users:read",  # Basic API access
            "openid",  # OpenID Connect
            "email",  # User email
            "profile",  # User profile
        ]

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": " ".join(scopes),
        }
        return f"{ASANA_AUTH_URL}?{urlencode(params)}"

    def _open_browser(self, url: str) -> None:
        """Open the default browser to the authentication URL"""
        if not webbrowser.open(url):
            print("Failed to open browser automatically")
            print(f"Please manually visit: {url}")

    def _exchange_code_for_tokens(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access and refresh tokens"""
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": redirect_uri,
            "code": code,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(ASANA_TOKEN_URL, data=data)
            resp.raise_for_status()
            return resp.json()

    def _run_server(self, port: int) -> None:
        """Run the FastAPI server in a separate thread"""
        try:
            app = self.create_app()
            config = uvicorn.Config(
                app=app,
                host="localhost",
                port=port,
                log_level="error",  # Suppress uvicorn logs
                access_log=False,
            )
            server = uvicorn.Server(config)
            self._server = server

            # Signal that server is ready
            self.server_started.set()

            # Run server (this blocks until shutdown)
            asyncio.run(server.serve())

        except Exception as e:
            self.error = f"Server error: {e}"
            self.server_started.set()
            self.auth_completed.set()

    def authenticate(self, timeout: int = 300) -> bool:
        """
        Perform the OAuth2 authentication flow

        Args:
            timeout: Maximum time to wait for authentication (seconds)

        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            port = 8765
            redirect_uri = f"http://localhost:{port}/callback"
            print(f"✓ Starting local callback server on port {port}")

            # Generate state for CSRF protection
            self.state = secrets.token_urlsafe(16)

            # Start server in background thread
            self._server_thread = threading.Thread(target=self._run_server, args=(port,), daemon=True)
            self._server_thread.start()

            # Wait for server to start
            if not self.server_started.wait(timeout=10):
                raise Exception("Server failed to start within 10 seconds")

            if self.error:
                raise Exception(self.error)

            # Build auth URL and open browser
            auth_url = self._build_auth_url(redirect_uri, self.state)
            print("✓ Opening browser for authorization...")
            print(f"\nIf browser doesn't open, visit:\n{auth_url}\n")
            self._open_browser(auth_url)

            # Wait for authentication to complete
            print("Waiting for authorization...")
            if not self.auth_completed.wait(timeout=timeout):
                raise Exception(f"Authentication timed out after {timeout} seconds")

            # Ensure server is shutting down
            self._request_shutdown()
            if self._server_thread and self._server_thread.is_alive():
                if self._server:
                    asyncio.run(self._server.shutdown())
                # Wait for the thread to terminate
                self._server_thread.join()
            # Check for errors
            if self.error:
                raise Exception(self.error)

            if not self.code:
                raise Exception("No authorization code received")

            # Exchange code for tokens
            print("✓ Authorization code received")
            print("✓ Exchanging code for tokens...")

            tokens = self._exchange_code_for_tokens(self.code, redirect_uri)
            _LocalAsanaTokens.save_token(self.client_id, tokens)

            return True

        except Exception as e:
            print(f"\n✗ Authentication failed: {e}")
            return False

    def has_saved_tokens(self) -> bool:
        """Check if we have saved tokens for this client"""
        if self.yaml_file.exists():
            try:
                with open(self.yaml_file, "r") as f:
                    tokens = yaml.safe_load(f) or {}
                return self.client_id in tokens and bool(tokens[self.client_id].get("access_token"))
            except Exception:
                pass
        return False


def get_asana_client(app_name: str) -> httpx.Client:
    """Return an authenticated httpx.Client for Asana using the latest token for the requested app.
    - Loads app credentials from AWS Secrets Manager at secret name 'asana/atlas_app'.
    - Refreshes tokens if expired using refresh_token.
    """
    creds = get_secret(f"asana/{app_name}_app")
    client_id = creds["client_id"]
    client_secret = creds["client_secret"]
    record = _LocalAsanaTokens.load_token(client_id)
    if record is None:
        raise RuntimeError(f"No saved tokens for client_id {client_id}. Run atlas_login.py to authenticate.")

    now = int(time.time())
    expires_at = int(record.get("expires_at", 0))
    access_token = record.get("access_token")
    refresh_token = record.get("refresh_token")

    # Refresh if expired (or about to expire)
    if now >= expires_at and refresh_token:
        data = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(ASANA_TOKEN_URL, data=data)
            resp.raise_for_status()
            payload = resp.json()

        _LocalAsanaTokens.save_token(client_id, payload)
        record = _LocalAsanaTokens.load_token(client_id)
        if record is None:
            raise RuntimeError(f"No saved tokens for client_id {client_id}. Run atlas_login.py to authenticate.")
        access_token = record["access_token"]

    if not access_token:
        raise RuntimeError("Failed to acquire Asana access token. Re-run atlas_login.py")

    client = httpx.Client(
        base_url="https://app.asana.com/api/1.0",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30.0,
    )
    return client


def login(name: str, force: bool = False, timeout: int = 300) -> None:
    cli = AsanaOAuthCLI(name)

    if cli.has_saved_tokens() and not force:
        print(f"Found existing tokens for client {cli.client_id}")
        print("  Use --force to get new tokens")

    print(f"Authenticating with Asana (client: {cli.client_id})")
    if cli.authenticate(timeout=timeout):
        print("Authentication successful!")
    else:
        print("Authentication failed!")
