"""Configuration for GitHub webhook service."""

import json
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for GitHub webhook service."""

    GITHUB_WEBHOOK_SECRET: str | None = None
    ASANA_PAT: str | None = None
    ASANA_CLIENT_ID: str | None = None
    ASANA_CLIENT_SECRET: str | None = None
    ASANA_WORKSPACE_GID: str | None = None
    ASANA_PROJECT_GID: str | None = None
    ASANA_ROSTER_PROJECT_GID: str = Field(default="1209948553419016")
    ASANA_GH_LOGIN_FIELD_GID: str = Field(default="1210594297567963")
    ASANA_EMAIL_FIELD_GID: str = Field(default="1209050603577235")
    ASANA_GITHUB_URL_FIELD_ID: str | None = None
    ASANA_RETRY_MAX_ATTEMPTS: int = 3
    ASANA_RETRY_INITIAL_DELAY_MS: float = 500.0
    ASANA_RETRY_MAX_DELAY_MS: float = 8000.0
    GITHUB_TOKEN: str | None = None

    # AWS Secrets Manager integration
    AWS_REGION: str = Field(default="us-east-1")
    USE_AWS_SECRETS: bool = Field(default=False)

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.USE_AWS_SECRETS:
            self._load_from_aws_secrets()

    def _load_from_aws_secrets(self):
        """Load secrets from AWS Secrets Manager."""
        try:
            import boto3
            from botocore.exceptions import ClientError

            client = boto3.client("secretsmanager", region_name=self.AWS_REGION)

            # Load GitHub webhook secret
            if not self.GITHUB_WEBHOOK_SECRET:
                try:
                    response = client.get_secret_value(SecretId="github/webhook-secret")
                    self.GITHUB_WEBHOOK_SECRET = response["SecretString"]
                except ClientError:
                    pass

            # Load Asana credentials - prefer atlas_app (client_id/client_secret) for service auth
            if not self.ASANA_CLIENT_ID or not self.ASANA_CLIENT_SECRET:
                try:
                    response = client.get_secret_value(SecretId="asana/atlas_app")
                    atlas_app_data = json.loads(response["SecretString"])
                    self.ASANA_CLIENT_ID = atlas_app_data.get("client_id")
                    self.ASANA_CLIENT_SECRET = atlas_app_data.get("client_secret")
                except (ClientError, json.JSONDecodeError, KeyError):
                    pass

            # Fallback to PAT if OAuth credentials not available
            if not self.ASANA_PAT:
                try:
                    response = client.get_secret_value(SecretId="asana/api-key")
                    self.ASANA_PAT = response["SecretString"]
                except ClientError:
                    try:
                        response = client.get_secret_value(SecretId="asana/access-token")
                        self.ASANA_PAT = response["SecretString"]
                    except ClientError:
                        pass

            if not self.ASANA_WORKSPACE_GID:
                try:
                    response = client.get_secret_value(SecretId="asana/workspace-gid")
                    self.ASANA_WORKSPACE_GID = response["SecretString"]
                except ClientError:
                    pass

            # Note: asana/bugs-project-gid might be different from ASANA_PROJECT_GID
            # You may need to create a separate secret or use the existing one
            if not self.ASANA_PROJECT_GID:
                try:
                    response = client.get_secret_value(SecretId="asana/bugs-project-gid")
                    self.ASANA_PROJECT_GID = response["SecretString"]
                except ClientError:
                    pass

            # Load GitHub token for PR description updates
            if not self.GITHUB_TOKEN:
                try:
                    response = client.get_secret_value(SecretId="github/token")
                    self.GITHUB_TOKEN = response["SecretString"]
                except ClientError:
                    pass

        except ImportError:
            pass
        except Exception:
            pass


settings = Settings()
