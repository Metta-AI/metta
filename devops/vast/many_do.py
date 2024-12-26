#!/usr/bin/env python3

import argparse
import subprocess
import time
import sys
import os
import json
from vastai import VastAI
import pandas as pd
from io import StringIO
import re
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_iterator_input(input_str):
    """Parse user input to generate a list of iterators."""
    iterators = []
    parts = input_str.split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                iterators.extend(range(start, end+1))
            except ValueError:
                print(f"Invalid range input: {part}")
                sys.exit(1)
        else:
            try:
                iterators.append(int(part))
            except ValueError:
                print(f"Invalid input: {part}")
                sys.exit(1)
    return sorted(iterators)

def get_instance_labels(base_name, iterators):
    """Generate instance labels from base name and iterators."""
    return [f"{base_name}-{i}" for i in iterators]

def wait_for_ready(label):
    """Wait for a machine with the given label to become ready."""
    cmd = ["python3", "devops/vast/do.py", "wait_for_ready", label]
    subprocess.run(cmd, check=True)

def attach_ssh_key(vastai, instance_id):
    """Attach the user's SSH key to the instance."""
    ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    try:
        with open(ssh_key_path) as f:
            # ssh_key = f.read().strip()
            ssh_key = f.read()
            # vastai = VastAI(api_key='5a9b0385999d95172ff3ab0b6f45ad4c1ae79fb36939804fe0032b6c806a1ef2')
            print(f"Instance ID: {instance_id}")
            print(f"ssh_key: {ssh_key}")
            vastai.attach_ssh(instance_id, ssh_key)
            return True
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading SSH key from {ssh_key_path}: {str(e)}")
        return False
    
def get_instance_id(data_str, target_label):
    """
    Extracts the ID associated with a given instance label using pandas with enhanced parsing.

    :param data_str: The raw string data containing instance details.
    :param target_label: The label of the instance whose ID is to be found.
    :return: The ID as an integer if found, else None.
    """
    # Preprocess the data string to ensure consistent spacing
    # Replace multiple spaces with a single space
    # Split the data into lines
    lines = data_str.strip().split('\n')
    
    if not lines:
        print("Error: No data provided.")
        return None

    # Process the header line to replace spaces with underscores for multi-word column names
    header = lines[0]
    # Define replacements for multi-word columns
    replacements = {
        'Util. %': 'Util_%',
        'SSH Addr': 'SSH_Addr',
        'SSH Port': 'SSH_Port',
        'Net up': 'Net_up',
        'Net down': 'Net_down',
        'age(hours)': 'age_hours'
    }
    for original, new in replacements.items():
        header = header.replace(original, new)
    
    # Replace the header line in the lines list
    lines[0] = header
    
    # Join the lines back into a single string
    cleaned_data = '\n'.join(lines)
    print(f"cleaned_data: {cleaned_data}\n")
    
    # Use StringIO to simulate a file-like object for pandas
    data_io = StringIO(cleaned_data)
    print(f"data_io: {data_io}\n")
    
    # Read the data into a DataFrame with corrected header
    try:
        df = pd.read_csv(
            data_io, 
            sep=r'\s+',                # Use regex to handle multiple spaces
            engine='python',           # Use Python engine for regex separator
            na_values='-',             # Treat '-' as NaN
            dtype=str                  # Read all columns as strings to prevent type issues
        )
    except Exception as e:
        print("Error parsing data:", e)
        return None
    
    # Debug: Print DataFrame columns and first few rows
    print("DataFrame Columns:", df.columns.tolist())
    print("DataFrame Head:\n", df.head(), "\n")
    
    # Check if 'Label' column exists
    if 'Label' not in df.columns:
        print("Error: 'Label' column not found in DataFrame.")
        return None
    
    # Search for the target label
    matched_rows = df[df['Label'] == target_label]
    
    if not matched_rows.empty:
        # Assuming IDs are numeric, convert to integer
        try:
            instance_id = int(matched_rows.iloc[0]['ID'])
            return instance_id
        except ValueError:
            print(f"Error: ID '{matched_rows.iloc[0]['ID']}' is not a valid integer.")
            return None
    else:
        return None

def rent_instances(instances, rent_args=None):
    """Rent specified instances."""
    rent_args = rent_args or []
    rented_instances = []
    
    # First, attempt to rent all instances
    for label in instances:
        print(f"Attempting to rent instance: {label}")
        rent_command = ["python3", "devops/vast/do.py", "rent", label] + rent_args
        result = subprocess.run(rent_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Successfully rented instance: {label}")
            rented_instances.append(label)
        else:
            print(f"Failed to rent instance: {label}")
            print(f"Error: {result.stderr}")
        time.sleep(2)
    
    if rented_instances:
        print("\nWaiting for instances to become ready and attaching SSH keys...")
        wait_and_setup_instances(rented_instances)
    
    return rented_instances

def wait_and_setup_instances(instances):
    """Wait for instances to become ready and attach SSH keys in parallel."""
    ready_instances = set()
    max_attempts = 30  # 5 minutes maximum wait time
    attempt = 0
    
    # First wait for all instances to be ready
    remaining_instances = instances.copy()
    while remaining_instances and attempt < max_attempts:
        for label in list(remaining_instances):  # Create a copy of the list for iteration
            try:
                cmd = ["python3", "devops/vast/do.py", "wait_for_ready", label]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if result.returncode == 0:
                    print(f"Instance {label} is ready")
                    remaining_instances.remove(label)
                    ready_instances.add(label)
            except subprocess.CalledProcessError as e:
                print(f"Error checking ready status for {label}: {str(e)}")
        
        if remaining_instances:
            print(f"Waiting for {len(remaining_instances)} instances to become ready... "
                  f"({', '.join(remaining_instances)})")
            time.sleep(10)
            attempt += 1
    
    if remaining_instances:
        print(f"Warning: Timed out waiting for instances: {', '.join(remaining_instances)}")
        return ready_instances

    # Now attach SSH keys to all ready instances
    print("\nAttaching SSH keys to ready instances...")
    vastai = VastAI(api_key='5a9b0385999d95172ff3ab0b6f45ad4c1ae79fb36939804fe0032b6c806a1ef2')
    instances = vastai.show_instances()
    for label in ready_instances:
        try:
            cmd = ["python3", "devops/vast/do.py", "get-ssh-details", label]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                instance_id = get_instance_id(instances, label)
                if instance_id:
                    print(f"Attaching SSH key to instance {label}...")
                    if attach_ssh_key(vastai, instance_id):
                        print(f"Successfully attached SSH key to instance {label}")
                    else:
                        print(f"Failed to attach SSH key to instance {label}")
                else:
                    print(f"Could not find instance ID for label {label}")
        except subprocess.CalledProcessError as e:
            print(f"Error attaching SSH key for {label}: {str(e)}")
    
    return ready_instances

def setup_instance(label, max_retries=3):
    """Setup a single instance with retries."""
    print(f"Starting setup for instance: {label}")
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        print(f"[{label}] Setup attempt {attempt} of {max_retries}")
        with open(f"setup_{label}.log", "a") as log_file:
            setup_command = ["python3", "devops/vast/do.py", "setup", label]
            process = subprocess.Popen(setup_command, stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()
        if process.returncode == 0:
            print(f"Setup completed for instance: {label} on attempt {attempt}")
            return True
        else:
            print(f"Setup failed for instance: {label} on attempt {attempt}.")
            print(f"Check setup_{label}.log for details.")
            if attempt < max_retries:
                print(f"Retrying setup for instance: {label}...")
    print(f"Setup ultimately failed for instance: {label} after {max_retries} attempts.")
    return False

def setup_instances(instances, max_retries=3):
    """Run setup on specified instances in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(setup_instance, label, max_retries): label for label in instances}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Setup encountered an exception for instance {label}: {e}")

def parallel_rsync(label):
    """Helper function to run rsync for a single instance."""
    print(f"Starting rsync for instance: {label}")
    rsync_command = ["python3", "devops/vast/do.py", "rsync", label]
    subprocess.run(rsync_command)

def rsync_instances(instances):
    """Run rsync on specified instances in parallel."""
    print("Starting parallel rsync...")

    # Adjust processes=4 to suit your environment or system CPU count
    with Pool(processes=4) as pool:
        pool.map(parallel_rsync, instances)

def execute_command(instances, command_template):
    """Execute a command on specified instances."""
    for label in instances:
        iterator = int(label.split('-')[-1])
        actual_command = command_template.format(iterator=iterator)
        print(f"Executing command on instance: {label}")
        run_remote_command(label, actual_command)

def run_remote_command(label, command):
    """Run a command on a remote instance inside a named screen session."""
    ssh_details_command = ["python3", "devops/vast/do.py", "get-ssh-details", label]
    result = subprocess.run(ssh_details_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Failed to get SSH details for instance {label}.")
        print(f"Error: {result.stderr}")
        return
    
    # Ensure the output is split correctly
    ssh_details = result.stdout.strip().split()
    if len(ssh_details) != 2:
        print(f"Unexpected SSH details format for instance {label}: {result.stdout}")
        return
    
    ssh_host, ssh_port = ssh_details

    # Create or reuse a named screen session (named train_<label>).
    full_command = (
        f"screen -dmS train_{label} bash -c \"cd /workspace/metta && {command}\""
    )

    ssh_command = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host} '{full_command}'"
    subprocess.run(ssh_command, shell=True)

def kill_instances(instances):
    """Kill specified instances."""
    for label in instances:
        print(f"Killing instance: {label}")
        kill_command = ["python3", "devops/vast/do.py", "kill", label]
        result = subprocess.run(kill_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Successfully killed instance: {label}")
        else:
            print(f"Failed to kill instance: {label}")
            print(f"Error: {result.stderr}")

def main():
    parser = argparse.ArgumentParser(description='Manage multiple Vast.ai instances')
    parser.add_argument('--base-name', type=str, required=True, help='Base name for instances (e.g., alex.12.24.a)')
    parser.add_argument('--iterators', type=str, required=True, help='Instance iterators (e.g., "1-6" or "2,3,6")')
    parser.add_argument('--action', type=str, required=True, choices=['rent', 'setup', 'rsync', 'execute', 'kill'],
                      help='Action to perform on instances')
    parser.add_argument('--max-dph', type=str, help='Maximum daily price in dollars (for rent action)')
    parser.add_argument('--command', type=str, help='Command template to execute (for execute action). Use {iterator} as placeholder')

    args = parser.parse_args()

    iterators = parse_iterator_input(args.iterators)
    instances = get_instance_labels(args.base_name, iterators)

    print(f"Selected instances: {', '.join(instances)}")

    if args.action == 'rent':
        rent_args = []
        if args.max_dph:
            rent_args.extend(['--max-dph', args.max_dph])
        rent_instances(instances, rent_args)
    elif args.action == 'setup':
        setup_instances(instances)
    elif args.action == 'rsync':
        rsync_instances(instances)
    elif args.action == 'execute':
        if not args.command:
            print("Error: --command is required for execute action")
            sys.exit(1)
        execute_command(instances, args.command)
    elif args.action == 'kill':
        kill_instances(instances)

if __name__ == '__main__':
    main() 