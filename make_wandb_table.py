import concurrent.futures
import re
import sys
import time

import pandas as pd
import wandb

ENTITY = "metta-research"
PROJECT = "metta"


def parse_gpus_from_name(name):
    # Look for pattern like "1x4", "2x8"
    match = re.search(r"(\d+)x(\d+)", name)
    if match:
        nodes = int(match.group(1))
        gpus_per_node = int(match.group(2))
        return nodes * gpus_per_node
    return None


def parse_unroll_from_name(name):
    # ml_1 -> 1
    match = re.search(r"ml_(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def parse_layers_from_name(name):
    # l_1 -> 1
    match = re.search(r"\.l_(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def get_config_val_nested(config, path):
    # Try direct key first (flattened)
    if path in config:
        return config[path]

    # Try nested
    keys = path.split(".")
    curr = config
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return None
    return curr


def get_config_any_path(config, suffix_path):
    # Try generic paths: suffix_path, TrainTool.suffix_path
    val = get_config_val_nested(config, suffix_path)
    if val is not None:
        return val
    val = get_config_val_nested(config, f"TrainTool.{suffix_path}")
    if val is not None:
        return val
    return None


def process_run(run):
    name = run.name
    # Accessing run.config might trigger lazy loading if not already present
    config = run.config

    # GPUs
    gpus = parse_gpus_from_name(name)

    # Muesli
    muesli_enabled = get_config_any_path(config, "trainer.losses.muesli_model.enabled")
    if muesli_enabled is None:
        muesli_enabled = False

    # Unroll Steps
    unroll = get_config_any_path(config, "trainer.losses.dynamics.unroll_steps")
    if unroll is None:
        unroll = parse_unroll_from_name(name)

    # Core Resnet Layers
    layers = parse_layers_from_name(name)
    if layers is None:
        layers = get_config_any_path(config, "policy_architecture.core_resnet_layers")

    return {
        "Run Name": name,
        "Total GPUs": gpus,
        "Muesli": muesli_enabled,
        "Unroll Steps": unroll,
        "Core Resnet Layers": layers,
    }


print(f"Connecting to WandB ({ENTITY}/{PROJECT})...")
api = wandb.Api(timeout=30)

print("Fetching runs starting with 'daveey.'...")
# We use the iterator to avoid fetching everything at once if possible,
# but to parallelize efficiently we might want to materialize the list.
# Let's fetch the list first.
try:
    runs = api.runs(f"{ENTITY}/{PROJECT}", {"name": {"$regex": "^daveey\\."}})
    # Converting to list forces fetching all pages
    all_runs = list(runs)
    print(f"Found {len(all_runs)} runs.")
except Exception as e:
    print(f"Error fetching runs: {e}")
    sys.exit(1)

print(f"Processing {len(all_runs)} runs with thread pool...")

data = []
start_time = time.time()

# Use ThreadPoolExecutor to process runs in parallel (speeds up if network calls needed for config)
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process_run, run): run for run in all_runs}

    completed = 0
    total = len(all_runs)

    for future in concurrent.futures.as_completed(futures):
        completed += 1
        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"\rProcessing: {completed}/{total} ({rate:.1f} runs/sec)", end="", flush=True)

        try:
            result = future.result()
            data.append(result)
        except Exception:
            # run = futures[future]
            # print(f"\nError processing run {run.name}: {e}")
            pass

print("\nDone processing.")

df = pd.DataFrame(data)
if not df.empty:
    df = df.sort_values("Run Name")
    print(df.to_string(index=False))
else:
    print("No data found.")
