#!/usr/bin/env python3
"""Test atlas_app credentials for Asana service authentication."""

import os
import sys

sys.path.insert(0, "src")

os.environ["USE_AWS_SECRETS"] = "true"
os.environ["AWS_REGION"] = "us-east-1"

from github_webhook.asana_integration import _get_asana_access_token
from github_webhook.config import settings

print("Testing Asana authentication with atlas_app credentials...")
print(f"ASANA_CLIENT_ID: {settings.ASANA_CLIENT_ID is not None}")
print(f"ASANA_CLIENT_SECRET: {settings.ASANA_CLIENT_SECRET is not None}")
print(f"ASANA_PAT (fallback): {settings.ASANA_PAT is not None}")

try:
    token = _get_asana_access_token()
    print(f"\n✅ Successfully obtained access token (length: {len(token)})")
    print(f"Token preview: {token[:20]}...{token[-10:]}")
except Exception as e:
    print(f"\n❌ Failed to get access token: {e}")
