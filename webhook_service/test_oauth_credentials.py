#!/usr/bin/env python3
"""Test OAuth client credentials for Asana."""

import sys
sys.path.insert(0, 'src')
import os
os.environ['USE_AWS_SECRETS'] = 'true'
os.environ['AWS_REGION'] = 'us-east-1'

from github_webhook.config import settings

print("Checking OAuth credentials...")
print(f"ASANA_CLIENT_ID: {settings.ASANA_CLIENT_ID is not None}")
print(f"ASANA_CLIENT_SECRET: {settings.ASANA_CLIENT_SECRET is not None}")
print(f"ASANA_PAT (fallback): {settings.ASANA_PAT is not None}")

if settings.ASANA_CLIENT_ID and settings.ASANA_CLIENT_SECRET:
    print("\n✅ OAuth credentials found! Testing token exchange...")
    try:
        from github_webhook.asana_integration import _get_asana_access_token
        token = _get_asana_access_token()
        print(f"✅ Successfully obtained access token (length: {len(token)})")
    except Exception as e:
        print(f"❌ Failed to get access token: {e}")
elif settings.ASANA_PAT:
    print("\n⚠️  Using PAT fallback (OAuth credentials not found)")
else:
    print("\n❌ No authentication credentials found")
