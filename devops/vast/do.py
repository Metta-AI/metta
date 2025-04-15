#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time

def gen_search_cmd(args):
    """Generate the search command that matches the given criteria."""
    cmd_str = f'vastai search offers \
        "num_gpus={args.num_gpus}" \
        "cpu_cores_effective>{args.min_cpu_cores}" \
        "inet_down>{args.min_inet}" \
        "inet_up>{args.min_inet}" \
        "cuda_vers>={args.min_cuda}" \
        "geolocation={args.geo}" \
        "gpu_name={args.gpu_name}" \
        "rented=False" \
        "dph<{args.max_dph}" \
        -o dph-'
    return cmd_str


def search_command(args):
    """Search for machines that match the given criteria."""
    cmd_str = gen_search_cmd(args)
    subprocess.run(cmd_str, shell=True, check=True)


def rent_command(args):
    """Rent a machine with the given label."""
    print(f"Renting with label: {args.label}")
    cmd_str = gen_search_cmd(args)
    print(cmd_str)

    output = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
    try:
        server_id = output.stdout.splitlines()[1].split()[0]
    except IndexError:
        quit("No server found")

    cmd_str = f"vastai create instance {server_id} \
        --image {args.image} \
        --disk 60 \
        --onstart-cmd '/bin/bash' \
        --label {args.label} \
        --ssh --direct \
        --args --ulimit nofile=unlimited --ulimit nproc=unlimited"
    subprocess.run(cmd_str, shell=True, check=True)


def label_to_instance(label):
    """Get the instance from the label."""
    output = subprocess.run("vastai show instances --raw", shell=True, check=True, capture_output=True, text=True)
    instances = json.loads(output.stdout)
    for instance in instances:
        if instance["label"] == label:
            return instance
    raise Exception(f"Instance {label} not found")


def label_to_id(label):
    """Get the instance ID from the label."""
    return label_to_instance(label)["id"]


def kill_command(args):
    """Destroy a machine with the given label."""
    cmd_str = f"vastai destroy instance {label_to_id(args.label)}"
    subprocess.run(cmd_str, shell=True, check=True)


def get_str(dict, value):
    """Get display string even if missing or None."""
    return str(dict.get(value)).strip()


def show_command(args):
    """Show all current instances."""
    cmd_str = "vastai show instances --raw"
    output = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
    instances = json.loads(output.stdout)
    for instance in instances:
        print(
            f"{instance['label']}"
            f" status:{get_str(instance, 'actual_status')}"
            f" CPU:{get_str(instance, 'cpu_util')}%"
            f" GPU:{get_str(instance, 'gpu_util')}%"
            f" [{get_str(instance, 'status_msg')}]"
            f" root@{instance['ssh_host']}:{instance['ssh_port']}"
        )


def wait_for_ready(label):
    """Wait for a machine with the given label to become ready."""
    instance = label_to_instance(label)
    # Wait for the instance to become ready.
    while instance["actual_status"] != "running":
        print(f"Waiting for instance {label} to become ready... ({instance['actual_status']})")
        time.sleep(5)
        instance = label_to_instance(label)


def wait_for_ssh(label):
    # Wait for the SSH key on the server to be ready.
    instance = label_to_instance(label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]
    cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} 'echo 1'"
    while True:
        try:
            subprocess.run(cmd, shell=True, check=True)
            break
        except Exception as e:
            print(f"Waiting for instance {label} to become ready... {e}")
            time.sleep(5)


def ssh_command(args):
    """SSH into a machine with the given label."""
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]
    cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host}"
    subprocess.run(cmd, shell=True, check=True)


def screen_command(args):
    """Like SSH, but starts or attaches to a screen session"""
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]
    cmd = f"ssh -t -o \
      StrictHostKeyChecking=no -p {ssh_port} \
      root@{ssh_host} \
      'cd /workspace/metta && screen -Rq'"
    print("Use ^A-D to detach from screen session.")
    subprocess.run(cmd, shell=True, check=True)


def tmux_command(args):
    """Like SSH, but starts or attaches to a tmux session"""
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]
    cmd = f"ssh -t -o \
      StrictHostKeyChecking=no -p {ssh_port} \
      root@{ssh_host} \
      'cd /workspace/metta && tmux new-session -A -s metta'"
    print("Use ^B-D to detach from tmux session.")
    subprocess.run(cmd, shell=True, check=True)


def setup_command(args):
    """Setup the machine for training."""
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]

    # Add local SSH key.
    ssh_keys = [
        "id_rsa.pub",
        "id_ed25519.pub",
    ]
    for key in ssh_keys:
        key_path = os.path.expanduser(f"~/.ssh/{key}")
        if os.path.exists(key_path):
            print(f"Adding ssh key {key_path}")
            ssh_key = open(key_path).read()
            cmd_attach = f"vastai attach ssh {instance['id']} '" + ssh_key + "'"
            subprocess.run(cmd_attach, shell=True, check=True)

    wait_for_ssh(args.label)

    cmd_setup = ["cd /workspace/metta", "git config --global --add safe.directory /workspace/metta"]
    if args.clean or args.branch != "main":
        cmd_setup.append("git reset --hard")
        cmd_setup.append("git clean -fdx")
        cmd_setup.append(f"git fetch origin {args.branch}")
        cmd_setup.append(f"git checkout {args.branch}")
        cmd_setup.append("git pull")
    cmd_setup.extend(["pip install -r requirements.txt", "bash devops/setup_build.sh"])
    cmd = f"ssh -t -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} '{' && '.join(cmd_setup)}'"
    subprocess.run(cmd, shell=True, check=True)
    # Copy the .netrc file
    scp_cmd = f"scp -P {ssh_port} $HOME/.netrc root@{ssh_host}:/root/.netrc"
    subprocess.run(scp_cmd, shell=True, check=True)


def rsync_command(args):
    """Rsync a working directory to a machine."""
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance["ssh_host"]
    ssh_port = instance["ssh_port"]
    # Walk dirs and recursively rsync only .py, .pyd, .pyx .yaml, and .sh files.
    # Current folder to /workspace/metta.
    # Show files transferred.
    cmd = (
        f"rsync -avz -e 'ssh -p {ssh_port}' --progress "
        f"--include='*/' "  # Include all directories
        f"--include='**/*.py' --include='**/*.pyd' --include='**/*.pyx' "
        f"--include='**/*.yaml' --include='**/*.sh' --exclude='*' "
        f"./ root@{ssh_host}:/workspace/metta"
    )
    subprocess.run(cmd, shell=True, check=True)


def send_keys(process, keys):
    """Send keyboard input to a process"""
    process.stdin.write(f"{keys}\n".encode())
    process.stdin.flush()


def main():
    """Swiss Army Knife for vast.ai."""

    parser = argparse.ArgumentParser(description="VAST CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search for machines")
    search_parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    search_parser.add_argument("--min-cpu-cores", type=int, default=8, help="Minimum CPU cores")
    search_parser.add_argument("--min-inet", type=int, default=100, help="Minimum internet speed (up/down) in Mbps")
    search_parser.add_argument("--min-cuda", type=str, default="12.1", help="Minimum CUDA version")
    search_parser.add_argument("--geo", type=str, default="US", help="Geolocation")
    search_parser.add_argument("--gpu-name", type=str, default="RTX_4090", help="GPU name")
    search_parser.add_argument("--max-dph", type=int, default=10, help="Maximum daily price in dollars")

    rent_parser = subparsers.add_parser("rent", help="Rent a machine")
    rent_parser.add_argument("label", type=str, help="Label for the instance")
    rent_parser.add_argument("--image", type=str, default="mettaai/metta:latest", help="Image to use")
    rent_parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    rent_parser.add_argument("--min-cpu-cores", type=int, default=8, help="Minimum CPU cores")
    rent_parser.add_argument("--min-inet", type=int, default=100, help="Minimum internet speed (up/down) in Mbps")
    rent_parser.add_argument("--min-cuda", type=str, default="12.1", help="Minimum CUDA version")
    rent_parser.add_argument("--geo", type=str, default="US", help="Geolocation")
    rent_parser.add_argument("--gpu-name", type=str, default="RTX_4090", help="GPU name")
    rent_parser.add_argument("--max-dph", type=int, default=1, help="Maximum daily price in dollars")

    kill_parser = subparsers.add_parser("kill", help="Destroy a machine")
    kill_parser.add_argument("label", type=str, help="Instance ID")

    subparsers.add_parser("show", help="Show a machine")

    ssh_parser = subparsers.add_parser("ssh", help="SSH into a machine")
    ssh_parser.add_argument("label", type=str, help="Instance ID")

    screen_parser = subparsers.add_parser(
        "screen", help="SSH into machine and start or attach to existing screen session"
    )
    screen_parser.add_argument("label", type=str, help="Instance ID")

    tmux_parser = subparsers.add_parser(
        "tmux", help="SSH into machine and open start or attach to existing tmux session"
    )
    tmux_parser.add_argument("label", type=str, help="Instance ID")

    setup_parser = subparsers.add_parser("setup", help="Setup a machine")
    setup_parser.add_argument("label", type=str, help="Instance ID")
    setup_parser.add_argument("--clean", type=bool, default=True, help="Cleans the workspace before setup")
    setup_parser.add_argument("--branch", type=str, default="main", help="Git branch to checkout")

    rsync_parser = subparsers.add_parser("rsync", help="Rsync a working directory to a machine")
    rsync_parser.add_argument("label", type=str, help="Instance ID")

    args = parser.parse_args()
    if args.command == "search":
        search_command(args)
    elif args.command == "rent":
        rent_command(args)
    elif args.command == "kill":
        kill_command(args)
    elif args.command == "show":
        show_command(args)
    elif args.command == "ssh":
        ssh_command(args)
    elif args.command == "screen":
        screen_command(args)
    elif args.command == "tmux":
        tmux_command(args)
    elif args.command == "setup":
        setup_command(args)
    elif args.command == "rsync":
        rsync_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
