#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import os
os.environ['USE_AWS_SECRETS'] = 'true'
os.environ['AWS_REGION'] = 'us-east-1'
from github_webhook.config import settings
print(f"GITHUB_TOKEN configured: {settings.GITHUB_TOKEN is not None}")
if settings.GITHUB_TOKEN:
    print(f"Token length: {len(settings.GITHUB_TOKEN)}")
else:
    print("❌ GITHUB_TOKEN not found in AWS Secrets Manager (secret: github/token)")
