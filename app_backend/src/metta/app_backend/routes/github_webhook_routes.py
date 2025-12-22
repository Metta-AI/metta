"""GitHub webhook routes for receiving and processing GitHub events."""

import hashlib
import hmac
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status

from metta.app_backend.config import settings
from metta.app_backend.github.metrics import metrics
from metta.app_backend.github.pr_handler import handle_pull_request_event
from metta.app_backend.route_logger import timed_http_handler

logger = logging.getLogger(__name__)


def create_github_webhook_router() -> APIRouter:
    """Create a GitHub webhook router."""
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    def verify_github_signature(payload_body: bytes, signature_header: Optional[str]) -> bool:
        """Verify that the webhook request came from GitHub."""
        if not settings.GITHUB_WEBHOOK_SECRET:
            # If no secret is configured, skip verification (dev mode)
            logger.warning("GITHUB_WEBHOOK_SECRET not set - skipping signature verification")
            return True

        if not signature_header:
            return False

        # GitHub sends the signature as "sha256=<hash>"
        hash_algorithm, github_signature = signature_header.split("=")
        if hash_algorithm != "sha256":
            return False

        # Calculate expected signature
        mac = hmac.new(
            settings.GITHUB_WEBHOOK_SECRET.encode("utf-8"),
            msg=payload_body,
            digestmod=hashlib.sha256,
        )
        expected_signature = mac.hexdigest()

        # Compare signatures
        return hmac.compare_digest(expected_signature, github_signature)

    @router.post("/github")
    @timed_http_handler
    async def github_webhook(
        request: Request,
        x_github_event: Optional[str] = Header(None),
        x_github_delivery: Optional[str] = Header(None),
        x_hub_signature_256: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """
        Receive GitHub webhook events.

        This endpoint processes GitHub webhook events, currently supporting:
        - pull_request events (action: opened)

        Future phases will add support for additional actions and event types.
        """
        # Get raw body for signature verification
        body = await request.body()

        # Verify webhook signature
        if not verify_github_signature(body, x_hub_signature_256):
            logger.warning(f"Invalid webhook signature for delivery {x_github_delivery}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

        # Parse JSON payload
        payload = await request.json()

        logger.info(
            f"Received GitHub webhook: event={x_github_event}, delivery={x_github_delivery}, "
            f"action={payload.get('action')}"
        )

        # Route event to appropriate handler
        if x_github_event == "ping":
            logger.info(f"Received ping event from GitHub (delivery {x_github_delivery})")
            return {"status": "ok", "message": "pong"}

        if x_github_event == "pull_request":
            with metrics.timed("github_asana.webhook.latency_ms", {"event": "pull_request"}):
                result = await handle_pull_request_event(
                    payload=payload,
                    delivery_id=x_github_delivery,
                )
            return {"status": "ok", "result": result}

        # For unsupported events, log and return success
        logger.info(f"Ignoring unsupported event type: {x_github_event}")
        return {"status": "ok", "message": f"Event type {x_github_event} not handled"}

    return router
