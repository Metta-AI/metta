"""AWS Lambda handler for GitHub webhook service with AWS Secrets Manager integration."""

import logging
import os

# Enable AWS Secrets Manager integration
os.environ["USE_AWS_SECRETS"] = "true"

from mangum import Mangum

from github_webhook.app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create ASGI adapter for Lambda
handler = Mangum(app, lifespan="off")


