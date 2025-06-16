import argparse
import logging
import os
import signal
import threading
import time

logger = logging.getLogger(__name__)

# Global variable to cache the heartbeat file path
_heartbeat_file_path = None


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the file specified by HEARTBEAT_FILE env var."""
    global _heartbeat_file_path

    # Cache the file path on first call
    if _heartbeat_file_path is None:
        _heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")
        if _heartbeat_file_path:
            os.makedirs(os.path.dirname(_heartbeat_file_path), exist_ok=True)

    # Only write if we have a valid path
    if _heartbeat_file_path:
        try:
            with open(_heartbeat_file_path, "w") as f:
                f.write(str(time.time()))
        except Exception:
            # Silently ignore errors to avoid disrupting training
            pass


def start_heartbeat(file_path: str, interval: float = 60.0) -> threading.Thread:
    """Start a background thread to update the heartbeat file."""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def _beat() -> None:
        while True:
            try:
                with open(file_path, "w") as f:
                    f.write(str(time.time()))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to write heartbeat: %s", exc)
            time.sleep(interval)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return thread


def monitor_heartbeat(file_path: str, pid: int, timeout: float = 600.0, interval: float = 60.0) -> None:
    """Monitor the heartbeat file and terminate the process group if stale."""

    while True:
        time.sleep(interval)
        try:
            last = os.path.getmtime(file_path)
        except FileNotFoundError:
            last = 0.0
        if time.time() - last > timeout:
            logger.error("No heartbeat detected for %s seconds. Terminating job", timeout)
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
    hb.add_argument("--interval", type=float, default=60.0)

    mon = sub.add_parser("monitor")
    mon.add_argument("file")
    mon.add_argument("--pid", type=int, default=os.getpid())
    mon.add_argument("--timeout", type=float, default=600.0)
    mon.add_argument("--interval", type=float, default=60.0)

    args = parser.parse_args(argv)

    if args.cmd == "heartbeat":
        start_heartbeat(args.file, args.interval).join()
    elif args.cmd == "monitor":
        monitor_heartbeat(args.file, pid=args.pid, timeout=args.timeout, interval=args.interval)
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
