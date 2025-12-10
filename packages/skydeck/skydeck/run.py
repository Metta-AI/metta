"""CLI entry point for SkyDeck dashboard."""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SkyDeck Dashboard - Web-based SkyPilot experiment manager")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Database file path (default: ~/.skydeck/skydeck.db)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set environment variables for configuration
    import os

    if args.db_path:
        os.environ["SKYDECK_DB_PATH"] = args.db_path
    if args.poll_interval:
        os.environ["SKYDECK_POLL_INTERVAL"] = str(args.poll_interval)

    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not found. Please install it: uv pip install uvicorn")
        sys.exit(1)

    # Start the server
    logger.info(f"Starting SkyDeck Dashboard on http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        uvicorn.run(
            "skydeck.app:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
