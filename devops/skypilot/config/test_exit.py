# test_exit.py
import os
import sys
import time

node_index = int(os.environ.get("NODE_INDEX", 0))
rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = rank == 0

print(f"Node {rank} starting (is_master: {is_master})")

if node_index > 0:  # Non-head node exits with error
    print(f"Node {node_index} Rank {rank} exiting with error")
    sys.exit(1)
else:
    time_to_sleep = 30 + rank
    print(f"Node {node_index} Rank {rank} sleeping for {time_to_sleep} seconds")
    time.sleep(time_to_sleep)
    print(f"Node {node_index} Rank {rank}  exiting normally")
    sys.exit(0)
