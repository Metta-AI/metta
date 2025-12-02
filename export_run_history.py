import wandb
import pandas as pd
from pathlib import Path

RUN_PATH = "metta-research/metta/prashant.fixed.maps.seed0"
OUTPUT_DIR = Path("analysis/reward_timeseries/data/fixed_maps/seed0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = [
    "reward_stream/value",
    "reward_stream/sample_index",
    "metric/sps",
    "env_id",
    "task_label",
]

api = wandb.Api()
run = api.run(RUN_PATH)

print(f"Starting export from {RUN_PATH}...")

batch = 0
rows = []
BATCH_SIZE = 250  # small batches so we see progress

for idx, row in enumerate(run.scan_history(page_size=500, keys=KEYS)):
    if idx == 0:
        print("Fetched first row...")
    rows.append(row)
    if idx % 500 == 0:
        print(f"Fetched {idx} rows...")
    if len(rows) >= BATCH_SIZE:
        pd.DataFrame(rows).to_parquet(
            OUTPUT_DIR / f"reward_stream_{batch}.parquet", index=False
        )
        print(f"Wrote batch {batch} ({len(rows)} rows)")
        rows = []
        batch += 1

if rows:
    pd.DataFrame(rows).to_parquet(
        OUTPUT_DIR / f"reward_stream_{batch}.parquet", index=False
    )
    print(f"Wrote batch {batch} ({len(rows)} rows)")

print("Done.")
