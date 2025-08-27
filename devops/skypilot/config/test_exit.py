# test_exit.py
import os
import sys
import time

node_index = int(os.environ.get("SKYPILOT_NODE_RANK", 0))
is_master = node_index == 0

print(f"Node {node_index} starting (is_master: {is_master})")

if node_index > 0:  # Worker nodes
    print(f"Worker {node_index} will run indefinitely...")
    # Workers just keep running
    while True:
        time.sleep(5)
        print(f"Worker {node_index} still running...")
else:
    # Master node exits early
    time_to_sleep = 10
    print(f"Master node running for {time_to_sleep} seconds...")
    time.sleep(time_to_sleep)

    # Master exits with code 0 (simulating timeout)
    print("Master node timeout reached, exiting with code 0")
    sys.exit(0)
