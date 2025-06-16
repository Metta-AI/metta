import argparse
import logging
import os
import signal
import time

logger = logging.getLogger(__name__)


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the globally configured file path."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")

    if heartbeat_file_path:
        try:
            with open(heartbeat_file_path, "w") as f:
                f.write(str(time.time()))
                print(f"Heartbeat recorded at {time.time()}")
        except Exception as exc:
            logger.warning("Failed to write heartbeat: %s", exc)


def monitor_heartbeat(file_path: str, pid: int, timeout: float = 600.0, check_interval: float = 60.0) -> None:
    """Monitor the heartbeat file and terminate the process group if stale."""

    while True:
        time.sleep(check_interval)
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
