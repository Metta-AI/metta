import argparse
import concurrent.futures
import logging
import os
import signal
import time

import wandb

logger = logging.getLogger(__name__)


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the globally configured file path."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")

    if heartbeat_file_path:
        try:
            with open(heartbeat_file_path, "w") as f:
                f.write(str(time.time()))
        except Exception as exc:
            logger.warning("Failed to write heartbeat: %s", exc)


# Default timeout for the alert sending operation itself
_ALERT_SEND_TIMEOUT_SECONDS = 30


def _send_wandb_alert_with_timeout(
    wandb_run: wandb.Run,
    title: str,
) -> None:
    """Send a W&B alert with a fixed internal timeout, providing only the title."""

    def send_alert() -> None:
        try:
            wandb_run.alert(title=title)
            logger.info("W&B alert sent with title: %s", title)
        except Exception as e:
            logger.warning("Failed to send W&B alert with title '%s': %s", title, e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send_alert)
        try:
            future.result(timeout=_ALERT_SEND_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning("W&B alert '%s' timed out after %s seconds.", title, _ALERT_SEND_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning("Exception during W&B alert '%s': %s", title, e)


def monitor_heartbeat(file_path: str, pid: int, timeout: float = 600.0, check_interval: float = 60.0) -> None:
    """Monitor the heartbeat file and terminate the process group if stale."""

    wandb_run: wandb.Run | None = wandb.run

    while True:
        time.sleep(check_interval)
        try:
            last = os.path.getmtime(file_path)
        except FileNotFoundError:
            last = 0.0
        if time.time() - last > timeout:
            logger.error("No heartbeat detected for %s seconds. Terminating job", timeout)

            if wandb_run:
                _send_wandb_alert_with_timeout(
                    wandb_run=wandb_run,
                    title="Heartbeat Timeout. Job terminated.",
                )
            else:
                logger.warning("W&B run not available, skipping alert for heartbeat timeout (PID: %s).", pid)

            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(10)
            os.killpg(pid, signal.SIGKILL)
            break


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    hb = sub.add_parser("heartbeat")
    hb.add_argument("file")

    mon = sub.add_parser("monitor")
    mon.add_argument("file")
    mon.add_argument("--pid", type=int, default=os.getpid())
    mon.add_argument("--timeout", type=float, default=600.0)
    mon.add_argument("--interval", type=float, default=60.0)

    args = parser.parse_args(argv)

    if args.cmd == "heartbeat":
        record_heartbeat()
    elif args.cmd == "monitor":
        monitor_heartbeat(args.file, pid=args.pid, timeout=args.timeout, check_interval=args.interval)
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
