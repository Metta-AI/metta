#!/usr/bin/env python3
"""Test if Asana token is working."""

import sys
sys.path.insert(0, 'src')
import os
os.environ['USE_AWS_SECRETS'] = 'true'
os.environ['AWS_REGION'] = 'us-east-1'

import asana
from github_webhook.config import settings

print(f"ASANA_PAT configured: {settings.ASANA_PAT is not None}")
print(f"ASANA_WORKSPACE_GID: {settings.ASANA_WORKSPACE_GID}")
print(f"ASANA_PROJECT_GID: {settings.ASANA_PROJECT_GID}")

if not settings.ASANA_PAT:
    print("❌ ASANA_PAT not configured")
    sys.exit(1)

try:
    configuration = asana.Configuration()
    configuration.access_token = settings.ASANA_PAT
    api_client = asana.ApiClient(configuration)
    users_api = asana.UsersApi(api_client)
    
    # Try to get current user
    result = users_api.get_user("me", {})
    print(f"✅ Asana token is valid! User: {result.get('name', 'Unknown')}")
except Exception as e:
    print(f"❌ Asana token test failed: {e}")
    sys.exit(1)
