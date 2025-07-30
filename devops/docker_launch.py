#!/usr/bin/env python3
"""
Docker-based sweep launcher that mimics SkyPilot's launch.py interface.
Usage: ./devops/docker_launch.py sweep <sweep_args> [--nodes N]

This script provides a simple interface to run distributed sweeps in Docker,
mimicking the SkyPilot environment but running locally.
"""

import subprocess
import sys
import threading
from pathlib import Path

# Configuration
COMPOSE_FILE = "metta/sweep/docker/docker-compose.yml"
CONTAINER_NAME = "metta-sweep-master"
WORKER_CONTAINER_NAME = "metta-sweep-worker"


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


def run_command_on_container(container_name, command, label):
    """Run a command on a specific container with labeled output."""
    cmd = ["docker", "exec", container_name, "bash", "-c", command]
    print(f"🚀 [{label}] Starting: {command}")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ [{label}] Completed successfully")
        else:
            print(f"❌ [{label}] Failed with exit code {result.returncode}")
        return result
    except Exception as e:
        print(f"❌ [{label}] Exception: {e}")
        return subprocess.CompletedProcess(cmd, 1)


def run_parallel_commands(containers_and_commands):
    """Run commands on multiple containers in parallel."""
    threads = []
    results = []

    def worker(container_name, command, label, results_list, index):
        result = run_command_on_container(container_name, command, label)
        results_list[index] = result

    # Prepare results list
    results = [None] * len(containers_and_commands)

    # Start all threads
    for i, (container_name, command, label) in enumerate(containers_and_commands):
        thread = threading.Thread(target=worker, args=(container_name, command, label, results, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Return the worst exit code
    exit_codes = [r.returncode for r in results if r is not None]
    return max(exit_codes) if exit_codes else 0


def ensure_environment():
    """Ensure Docker environment is running."""
    if not Path(COMPOSE_FILE).exists():
        print(f"❌ Docker setup not found at {COMPOSE_FILE}")
        print("💡 Run: ./devops/setup_docker_sandbox.sh")
        sys.exit(1)

    # Check if containers are running
    result = run_command(
        ["docker", "ps", "--filter", "name=metta-sweep", "--format", "{{.Names}}"],
        capture_output=True,
        check=False,
    )

    running_containers = result.stdout.strip().split("\n") if result.stdout.strip() else []

    if CONTAINER_NAME not in running_containers:
        print(f"❌ Master container {CONTAINER_NAME} is not running")
        print("💡 Run: ./devops/setup_docker_sandbox.sh")
        sys.exit(1)

    if WORKER_CONTAINER_NAME not in running_containers:
        print(f"❌ Worker container {WORKER_CONTAINER_NAME} is not running")
        print("💡 Run: ./devops/setup_docker_sandbox.sh")
        sys.exit(1)


def health_check():
    """Quick health check of the Docker environment."""
    # Simple check that both containers can execute commands
    master_result = run_command(
        ["docker", "exec", CONTAINER_NAME, "echo", "Master OK"],
        capture_output=True,
        check=False,
    )

    worker_result = run_command(
        ["docker", "exec", WORKER_CONTAINER_NAME, "echo", "Worker OK"],
        capture_output=True,
        check=False,
    )

    if master_result.returncode == 0 and worker_result.returncode == 0:
        print("✅ Both containers are healthy")
    else:
        print("⚠️  Some containers may not be ready")


def run_sweep(*args):
    """Run a sweep in the Docker environment."""
    ensure_environment()

    # Parse --nodes flag (handle both --nodes 2 and --nodes=2 formats)
    nodes = 1
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--nodes" and i + 1 < len(args):
            # Handle --nodes 2 format
            nodes = int(args[i + 1])
            i += 2  # Skip both --nodes and the value
        elif args[i].startswith("--nodes="):
            # Handle --nodes=2 format
            nodes = int(args[i].split("=", 1)[1])
            i += 1  # Skip just this argument
        else:
            filtered_args.append(args[i])
            i += 1

    # Build the command exactly like the working pattern
    args_str = " ".join(filtered_args)

    if nodes > 1:
        print(f"🔀 Distributed training: {nodes} nodes")
        print("🚀 Starting distributed sweep on both master and worker containers...")

        # Set environment variables for distributed training
        env_vars = [
            f"NUM_NODES={nodes}",
        ]
        env_str = " ".join(env_vars) + " "

        # Same command runs on both containers (SkyPilot-style)
        # Only add +hardware=macbook if it's not already present
        hardware_flag = "" if "+hardware=macbook" in args_str else " +hardware=macbook"
        sweep_cmd = (
            f"cd /home/metta/metta && source devops/setup.env && {env_str}devops/sweep.sh {args_str}{hardware_flag}"
        )

        print(f"🏃 Running sweep: {' '.join(filtered_args)}")
        print(f"🌐 Multi-node: {nodes} nodes")
        print(f"📍 Master container: {CONTAINER_NAME}")
        print(f"📍 Worker container: {WORKER_CONTAINER_NAME}")
        print()
        print("🔄 Executing same command on both containers simultaneously (SkyPilot-style)")
        print()

        # Execute the same command on both containers in parallel
        containers_and_commands = [
            (CONTAINER_NAME, sweep_cmd, "MASTER"),
            (WORKER_CONTAINER_NAME, sweep_cmd, "WORKER"),
        ]

        exit_code = run_parallel_commands(containers_and_commands)
        return subprocess.CompletedProcess([], exit_code)
    else:
        print("🔄 Single-node training")

        # Original single-node logic
        # Only add +hardware=macbook if it's not already present
        hardware_flag = "" if "+hardware=macbook" in args_str else " +hardware=macbook"
        sweep_cmd = f"cd /home/metta/metta && source devops/setup.env && devops/sweep.sh {args_str}{hardware_flag}"

        cmd = ["docker", "exec", CONTAINER_NAME, "bash", "-c", sweep_cmd]

        print(f"🏃 Running sweep: {' '.join(filtered_args)}")
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
        print("  sweep <args> [--nodes N]  - Run a sweep with specified arguments")
        print("  status                   - Check environment status")
        print("  logs [service]           - Show logs (optionally for specific service)")
        print("  stop                     - Stop the environment")
        print("  shell                    - Open shell in master container")
        print()
        print("Sweep Options:")
        print("  --nodes N               - Number of nodes for distributed training (default: 1)")
        print()
        print("Examples:")
        print("  ./devops/docker_launch.py sweep run=test trainer.total_timesteps=200")
        print("  ./devops/docker_launch.py sweep run=test --nodes 2 trainer.total_timesteps=200")
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
            print("Multi-node: ./devops/docker_launch.py sweep run=test --nodes 2 trainer.total_timesteps=200")
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
