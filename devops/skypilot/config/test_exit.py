# test_exit.py
import os
import sys
import time

node_index = int(os.environ.get("SKYPILOT_NODE_RANK", 0))
is_master = node_index == 0

print(f"Node {node_index} starting (is_master: {is_master})")

if node_index > 0:  # Worker nodes
    print(f"Worker {node_index} exiting immediately with code 0")
    sys.exit(0)
else:
    # Master node continues
    time_to_sleep = 10
    print(f"Master node running for {time_to_sleep} seconds...")
    time.sleep(time_to_sleep)

    # Master can set the final exit code
    print("Master node work complete, exiting with code 1")
    sys.exit(1)
