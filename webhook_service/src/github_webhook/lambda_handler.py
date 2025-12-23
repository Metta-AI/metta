"""AWS Lambda handler for GitHub webhook service."""

import logging

from mangum import Mangum

from github_webhook.app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create ASGI adapter for Lambda
handler = Mangum(app, lifespan="off")


