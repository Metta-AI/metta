"""GitHub webhook routes for receiving and processing GitHub events."""

import hashlib
import hmac
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status

from metta.app_backend.github_webhook.config import settings
from metta.app_backend.github_webhook.metrics import metrics
from metta.app_backend.github_webhook.pr_handler import handle_pull_request_event

logger = logging.getLogger(__name__)


def create_github_webhook_router() -> APIRouter:
    """Create a GitHub webhook router."""
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    def verify_github_signature(payload_body: bytes, signature_header: Optional[str]) -> bool:
        """Verify that the webhook request came from GitHub."""
        if not settings.GITHUB_WEBHOOK_SECRET:
            # In production (USE_AWS_SECRETS=true), the secret must be loaded
            # In dev mode (USE_AWS_SECRETS=false), we allow skipping verification
            if settings.USE_AWS_SECRETS:
                logger.error(
                    "GITHUB_WEBHOOK_SECRET not set but USE_AWS_SECRETS=true - "
                    "this indicates a configuration error. Rejecting request."
                )
                return False
            # Dev mode: allow skipping verification
            logger.warning("GITHUB_WEBHOOK_SECRET not set - skipping signature verification (dev mode)")
            return True

        if not signature_header or "=" not in signature_header:
            return False

        # GitHub sends the signature as "sha256=<hash>"
        hash_algorithm, github_signature = signature_header.split("=", 1)
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
    async def github_webhook(
        request: Request,
        x_github_event: Optional[str] = Header(None),
        x_github_delivery: Optional[str] = Header(None),
        x_hub_signature_256: Optional[str] = Header(None),
    ) -> Dict[str, Any]:
        """
        Receive GitHub webhook events.

        This endpoint processes GitHub webhook events, supporting:
        - ping events: returns pong
        - pull_request events with actions:
          * opened: create Asana task (with deduplication)
          * assigned/unassigned/edited: sync task assignee
          * closed: mark task complete (merged or not)
          * reopened: reopen task
          * synchronize: no-op (new commits pushed)
        - Other event types: acknowledged and ignored

        All pull_request actions return structured plan objects for observability.
        Unexpected errors are caught and logged without returning 500 to GitHub.
        """
        # Get raw body for signature verification
        body = await request.body()

        # Verify webhook signature
        if not verify_github_signature(body, x_hub_signature_256):
            logger.warning(
                f"Invalid webhook signature for delivery {x_github_delivery}. "
                f"Secret configured: {settings.GITHUB_WEBHOOK_SECRET is not None}, "
                f"Signature header present: {x_hub_signature_256 is not None}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse webhook payload: {e}", exc_info=True)
            return {"status": "error", "message": "Invalid JSON payload"}

        logger.info(
            f"Received GitHub webhook: event={x_github_event}, delivery={x_github_delivery}, "
            f"action={payload.get('action')}"
        )

        # Route event to appropriate handler
        if x_github_event == "ping":
            logger.info(f"Received ping event from GitHub (delivery {x_github_delivery})")
            return {"status": "ok", "message": "pong"}

        if x_github_event == "pull_request":
            try:
                with metrics.timed("github_asana.webhook.latency_ms", {"event": "pull_request"}):
                    result = await handle_pull_request_event(
                        payload=payload,
                        delivery_id=x_github_delivery,
                    )
                return {"status": "ok", "result": result}
            except Exception as e:
                logger.error(
                    f"Unexpected error processing pull_request event: {e}",
                    exc_info=True,
                    extra={"delivery": x_github_delivery, "action": payload.get("action")},
                )
                metrics.increment_counter("github_asana.dead_letter.count", {"operation": "webhook_handler"})
                return {
                    "status": "ok",
                    "result": {
                        "kind": "dead_letter",
                        "reason": "unexpected_error",
                        "error": str(e),
                    },
                }

        # For unsupported events, log and return success
        logger.info(f"Ignoring unsupported event type: {x_github_event}")
        return {"status": "ok", "message": f"Event type {x_github_event} not handled"}

    return router
