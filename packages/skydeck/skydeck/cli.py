"""CLI entry point for SkyDeck dashboard with daemon management."""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


def get_pid_file():
    """Get path to PID file."""
    return Path.home() / ".skydeck" / "skydeck.pid"


def get_log_file():
    """Get path to log file."""
    return Path.home() / ".skydeck" / "skydeck.log"


def is_running():
    """Check if skydeck server is running."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return False

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process is alive
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        # Process doesn't exist or invalid PID
        pid_file.unlink(missing_ok=True)
        return False


def get_pid():
    """Get PID of running server."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return None

    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def start_server(host="127.0.0.1", port=8000, restart=False):
    """Start the skydeck server."""
    # Ensure directory exists
    pid_file = get_pid_file()
    log_file = get_log_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if already running
    if is_running():
        if restart:
            print("Stopping existing server...")
            stop_server()
            time.sleep(1)
        else:
            print(f"SkyDeck is already running (PID: {get_pid()})")
            print(f"Dashboard: http://{host}:{port}")
            print("Use 'skydeck stop' to stop it or 'skydeck start --restart' to restart")
            return

    print("Starting SkyDeck dashboard...")

    # Start server as background process
    cmd = [
        sys.executable,
        "-m",
        "skydeck.run",
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Open log file
    with open(log_file, "w") as f:
        f.write("SkyDeck Dashboard Log\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'=' * 60}\n\n")

    # Start process
    with open(log_file, "a") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Write PID file
    with open(pid_file, "w") as f:
        f.write(str(process.pid))

    # Wait a moment and check if it started successfully
    time.sleep(2)

    if is_running():
        print(f"✓ SkyDeck started successfully (PID: {process.pid})")
        print(f"  Dashboard: http://{host}:{port}")
        print(f"  Logs: {log_file}")
        print("\nUse 'skydeck stop' to stop the server")
        print("Use 'skydeck' to open the dashboard in your browser")
    else:
        print("✗ Failed to start SkyDeck")
        print(f"Check logs: {log_file}")
        sys.exit(1)


def stop_server():
    """Stop the skydeck server."""
    if not is_running():
        print("SkyDeck is not running")
        return

    pid = get_pid()
    print(f"Stopping SkyDeck (PID: {pid})...")

    try:
        # Send SIGTERM
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit (up to 10 seconds)
        for _ in range(20):
            time.sleep(0.5)
            if not is_running():
                break

        # If still running, force kill
        if is_running():
            print("Process didn't stop gracefully, forcing...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        # Clean up PID file
        pid_file = get_pid_file()
        pid_file.unlink(missing_ok=True)

        print("✓ SkyDeck stopped")
    except Exception as e:
        print(f"Error stopping server: {e}")
        sys.exit(1)


def status_server():
    """Show status of skydeck server."""
    if is_running():
        pid = get_pid()
        print(f"SkyDeck is running (PID: {pid})")
        print("Dashboard: http://127.0.0.1:8000")
        print(f"Logs: {get_log_file()}")
    else:
        print("SkyDeck is not running")
        print("Use 'skydeck start' to start it")


def open_dashboard(host="127.0.0.1", port=8000):
    """Open dashboard in browser."""
    if not is_running():
        print("SkyDeck is not running, starting it...")
        start_server(host=host, port=port)
        time.sleep(2)

    url = f"http://{host}:{port}"
    print(f"Opening dashboard: {url}")
    webbrowser.open(url)


def show_logs():
    """Show server logs."""
    log_file = get_log_file()
    if not log_file.exists():
        print("No logs found")
        return

    print(f"Showing logs from: {log_file}")
    print("=" * 60)

    # Show last 50 lines
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line, end="")
    except Exception as e:
        print(f"Error reading logs: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SkyDeck Dashboard - Web-based SkyPilot experiment manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  skydeck              Open dashboard in browser (starts server if needed)
  skydeck start        Start the server
  skydeck stop         Stop the server
  skydeck restart      Restart the server
  skydeck status       Show server status
  skydeck logs         Show server logs

Examples:
  skydeck              # Open dashboard
  skydeck start        # Start server
  skydeck stop         # Stop server
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="open",
        choices=["start", "stop", "restart", "status", "logs", "open"],
        help="Command to execute (default: open)",
    )

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
        "--restart",
        action="store_true",
        help="Restart server if already running (use with 'start')",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Execute command
    if args.command == "start":
        start_server(host=args.host, port=args.port, restart=args.restart)
    elif args.command == "stop":
        stop_server()
    elif args.command == "restart":
        stop_server()
        time.sleep(1)
        start_server(host=args.host, port=args.port)
    elif args.command == "status":
        status_server()
    elif args.command == "logs":
        show_logs()
    elif args.command == "open":
        open_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
