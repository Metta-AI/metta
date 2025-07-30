#!/usr/bin/env python3
"""
Docker-based sweep launcher that mimics SkyPilot's launch.py interface.
Usage: ./devops/docker_launch.py sweep <sweep_args>

This script provides a simple interface to run distributed sweeps in Docker,
mimicking the SkyPilot environment but running locally.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
COMPOSE_FILE = "metta/sweep/docker/docker-compose.yml"
CONTAINER_NAME = "metta-sweep-master"


def run_command(cmd, check=True, capture_output=False, **kwargs):
    """Run a command with better error handling."""
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
        else:
            result = subprocess.run(cmd, check=check, **kwargs)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            sys.exit(e.returncode)
        return e


def ensure_environment():
    """Ensure Docker environment is running."""
    if not Path(COMPOSE_FILE).exists():
        print(f"❌ Docker setup not found at {COMPOSE_FILE}")
        print("💡 Run: ./devops/setup_docker_sandbox.sh")
        sys.exit(1)

    # Check if containers are running
    result = run_command(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True,
        check=False,
    )

    if CONTAINER_NAME not in result.stdout:
        print("❌ Docker environment is not running")
        print("💡 Run: ./devops/setup_docker_sandbox.sh")
        sys.exit(1)


def health_check():
    """Quick health check of the Docker environment."""
    result = run_command(
        ["docker", "exec", CONTAINER_NAME, "curl", "-s", "http://localhost:8080/health"],
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        print("✅ Environment is healthy")
    else:
        print("⚠️  Environment may not be fully ready (this might be okay)")


def run_sweep(*args):
    """Run a sweep in the Docker environment."""
    ensure_environment()

    # Build the command exactly like the working pattern
    args_str = " ".join(args)
    sweep_cmd = (
        f"cd /home/metta/metta && source .venv/bin/activate && "
        f"source devops/setup.env && devops/sweep.sh {args_str} +hardware=macbook"
    )

    cmd = ["docker", "exec", "-it", CONTAINER_NAME, "bash", "-c", sweep_cmd]

    print(f"🏃 Running sweep: {' '.join(args)}")
    print(f"📍 Container: {CONTAINER_NAME}")
    print()

    return run_command(cmd, check=False)


def status():
    """Check Docker environment status."""
    print("📊 Docker Environment Status:")
    print()

    result = run_command(["docker-compose", "-f", COMPOSE_FILE, "ps"], check=False)

    if result.returncode == 0:
        print()
        health_check()

    return result.returncode


def logs(service=None):
    """Show logs from the Docker environment."""
    cmd = ["docker-compose", "-f", COMPOSE_FILE, "logs"]

    if service:
        cmd.extend(["-f", service])
    else:
        cmd.append("-f")

    print(f"📜 Showing logs{f' for {service}' if service else ''}...")
    return run_command(cmd, check=False)


def stop():
    """Stop the Docker environment."""
    print("🛑 Stopping Docker environment...")
    return run_command(["docker-compose", "-f", COMPOSE_FILE, "down"], check=False)


def shell():
    """Open a shell in the master container."""
    ensure_environment()

    print(f"🐚 Opening shell in {CONTAINER_NAME}...")
    return run_command(["docker", "exec", "-it", CONTAINER_NAME, "bash"], check=False)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("🤖 Metta Docker Launcher")
        print()
        print("Usage:")
        print("  ./devops/docker_launch.py <command> [args...]")
        print()
        print("Commands:")
        print("  sweep <args>     - Run a sweep with specified arguments")
        print("  status          - Check environment status")
        print("  logs [service]  - Show logs (optionally for specific service)")
        print("  stop            - Stop the environment")
        print("  shell           - Open shell in master container")
        print()
        print("Examples:")
        print("  ./devops/docker_launch.py sweep run=test trainer.total_timesteps=200")
        print(
            "  ./devops/docker_launch.py sweep run=arena_test trainer.curriculum=/env/mettagrid/curriculum/arena/random"
        )
        print("  ./devops/docker_launch.py status")
        print("  ./devops/docker_launch.py shell")
        print()
        print("Setup:")
        print("  ./devops/setup_docker_sandbox.sh")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "sweep":
        if not args:
            print("❌ Sweep command requires arguments")
            print("Example: ./devops/docker_launch.py sweep run=test trainer.total_timesteps=200")
            sys.exit(1)
        return run_sweep(*args).returncode
    elif command == "status":
        return status()
    elif command == "logs":
        return logs(args[0] if args else None).returncode
    elif command == "stop":
        return stop().returncode
    elif command == "shell":
        return shell().returncode
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: sweep, status, logs, stop, shell")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
