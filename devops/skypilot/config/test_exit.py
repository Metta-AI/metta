# test_exit.py
import os
import sys
import time

node_index = int(os.environ.get("SKYPILOT_NODE_RANK", 0))

is_master = node_index == 0

print(f"Node {node_index} starting (is_master: {is_master})")

if node_index > 0:  # Non-head node exits with error
    print(f"Node {node_index} exiting with error")
    sys.exit(1)
else:
    time_to_sleep = 10
    print(f"Node {node_index} sleeping for {time_to_sleep} seconds")
    time.sleep(time_to_sleep)
    print(f"Node {node_index}  exiting normally")
    sys.exit(0)
