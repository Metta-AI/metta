#!/usr/bin/env python3
import argparse
import subprocess
import json
import time
import shlex
from typing import List

def gen_search_cmd(args):
    """ Generate the search command that matches the given criteria. """
    cmd_str = f'vastai search offers \
        num_gpus={args.num_gpus} \
        "cpu_cores_effective>{args.min_cpu_cores}" \
        "inet_down>{args.min_inet}" \
        "inet_up>{args.min_inet}" \
        "cuda_vers>={args.min_cuda}" \
        "geolocation={args.geo}" \
        gpu_name={args.gpu_name} \
        rented=False \
        "dph<{args.max_dph}" \
        -o dph-'
    return cmd_str

def search_command(args):
    """ Search for machines that match the given criteria. """
    cmd_str = gen_search_cmd(args)
    subprocess.run(cmd_str, shell=True, check=True)

def rent_command(args):
    """ Rent a machine with the given label. """
    print(f"Renting with label: {args.label}")
    cmd_str = gen_search_cmd(args)

    output = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
    server_id = output.stdout.splitlines()[1].split()[0]

    cmd_str = f"vastai create instance {server_id} \
        --image {args.image} \
        --disk 60 \
        --onstart-cmd '/bin/bash' \
        --label {args.label} \
        --ssh --direct \
        --args --ulimit nofile=unlimited --ulimit nproc=unlimited"
    subprocess.run(cmd_str, shell=True, check=True)

def label_to_instance(label):
    """ Get the instance from the label. """
    output = subprocess.run("vastai show instances --raw", shell=True, check=True, capture_output=True, text=True)
    instances = json.loads(output.stdout)
    for instance in instances:
        if instance['label'] == label:
            return instance
    raise Exception(f"Instance {label} not found")

def label_to_id(label):
    """ Get the instance ID from the label. """
    return label_to_instance(label)['id']

def kill_command(args):
    """ Destroy a machine with the given label. """
    cmd_str = f"vastai destroy instance {label_to_id(args.label)}"
    subprocess.run(cmd_str, shell=True, check=True)

def show_command(args):
    """ Show all current instances. """
    cmd_str = "vastai show instances --raw"
    output = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True)
    instances = json.loads(output.stdout)
    for instance in instances:
        statement = f"{instance['label']} status:{instance['actual_status']} CPU:{instance['cpu_util']}% GPU:{instance['gpu_util']}%"
        if instance['status_msg']:
            statement += f" [{instance['status_msg'].strip()}]"
        print(statement)

def wait_for_ready(label):
    """ Wait for a machine with the given label to become ready. """
    instance = label_to_instance(label)
    while instance['actual_status'] != 'running':
        print(f"Waiting for instance {label} to become ready... ({instance['actual_status']})")
        time.sleep(5)
        instance = label_to_instance(label)

def ssh_command(args):
    """ SSH into a machine with the given label. """
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance['ssh_host']
    ssh_port = instance['ssh_port']
    cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host}"
    subprocess.run(cmd, shell=True, check=True)

def screen_command(args):
    """ Like SSH, but starts or attaches to a screen session """
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance['ssh_host']
    ssh_port = instance['ssh_port']
    cmd = f"ssh -t -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} 'cd /workspace/metta && screen -Rq'"
    print("Use ^A-D to detach from screen session.")
    subprocess.run(cmd, shell=True, check=True)

def tmux_command(args):
    """ Like SSH, but starts or attaches to a tmux session """
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance['ssh_host']
    ssh_port = instance['ssh_port']
    cmd = f"ssh -t -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} 'cd /workspace/metta && tmux new-session -A -s metta'"
    print("Use ^B-D to detach from tmux session.")
    subprocess.run(cmd, shell=True, check=True)

def send_keys(process, keys):
    """Send keyboard input to a process"""
    process.stdin.write(f"{keys}\n".encode())
    process.stdin.flush()

def train_command(args):
    """ Train a model on a machine with the given label. """
    # THIS COMMAND IS STILL A WIP
    wait_for_ready(args.label)
    instance = label_to_instance(args.label)
    ssh_host = instance['ssh_host']
    ssh_port = instance['ssh_port']
    # Start the SSH session
    process = subprocess.Popen(
        f"ssh -t -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host}",
        stdin=subprocess.PIPE,
        shell=True
    )
    # Send commands sequentially
    send_keys(process, "cd /workspace/metta")
    send_keys(process, "screen -Rq")
    send_keys(process, "git pull")
    send_keys(process, "pip install -r requirements.txt")
    send_keys(process, "bash devops/setup_build.sh")
    # Run the train command
    send_keys(process, f"python -m tools.train --run={args.label} hardware=pufferbox wandb.enabled=true wandb.track=true {' '.join(args.train_args)}")

def main():
    """ Swiss Army Knife for vast.ai. """

    parser = argparse.ArgumentParser(description='VAST CLI tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    search_parser = subparsers.add_parser('search', help='Search for machines')
    search_parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    search_parser.add_argument('--min-cpu-cores', type=int, default=8, help='Minimum CPU cores')
    search_parser.add_argument('--min-inet', type=int, default=100, help='Minimum internet speed (up/down) in Mbps')
    search_parser.add_argument('--min-cuda', type=str, default='12.1', help='Minimum CUDA version')
    search_parser.add_argument('--geo', type=str, default='US', help='Geolocation')
    search_parser.add_argument('--gpu-name', type=str, default='RTX_4090', help='GPU model')
    search_parser.add_argument('--max-dph', type=float, default=1.0, help='Maximum dollars per hour')

    rent_parser = subparsers.add_parser('rent', help='Rent a machine')
    rent_parser.add_argument('label', type=str, help='Label for the instance')
    rent_parser.add_argument('--image', type=str, default='mettaai/metta:latest', help='Image to use')
    rent_parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    rent_parser.add_argument('--min-cpu-cores', type=int, default=8, help='Minimum CPU cores')
    rent_parser.add_argument('--min-inet', type=int, default=100, help='Minimum internet speed (up/down) in Mbps')
    rent_parser.add_argument('--min-cuda', type=str, default='12.1', help='Minimum CUDA version')
    rent_parser.add_argument('--geo', type=str, default='US', help='Geolocation')
    rent_parser.add_argument('--gpu-name', type=str, default='RTX_4090', help='GPU model')
    rent_parser.add_argument('--max-dph', type=float, default=1.0, help='Maximum dollars per hour')

    kill_parser = subparsers.add_parser('kill', help='Destroy a machine')
    kill_parser.add_argument('label', type=str, help='Instance ID')

    show_parser = subparsers.add_parser('show', help='Show a machine')

    ssh_parser = subparsers.add_parser('ssh', help='SSH into a machine')
    ssh_parser.add_argument('label', type=str, help='Instance ID')

    screen_parser = subparsers.add_parser('screen', help='SSH into machine and open start or attach to existing screen session')
    screen_parser.add_argument('label', type=str, help='Instance ID')

    tmux_parser = subparsers.add_parser('tmux', help='SSH into machine and open start or attach to existing tmux session')
    tmux_parser.add_argument('label', type=str, help='Instance ID')

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('label', type=str, help='Instance ID')
    train_parser.add_argument('train_args', nargs=argparse.REMAINDER, help='Arguments to pass to the train script')

    args = parser.parse_args()

    if args.command == 'search':
        search_command(args)
    elif args.command == 'rent':
        rent_command(args)
    elif args.command == 'kill':
        kill_command(args)
    elif args.command == 'show':
        show_command(args)
    elif args.command == 'ssh':
        ssh_command(args)
    elif args.command == 'screen':
        screen_command(args)
    elif args.command == 'tmux':
        tmux_command(args)
    elif args.command == 'train':
        train_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()