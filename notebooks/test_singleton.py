#!/usr/bin/env python3
"""Test the RunStore singleton pattern."""

from run_store import get_runstore

# Test 1: Verify singleton works
rs1 = get_runstore()
rs2 = get_runstore()
print(f"Singleton test: rs1 is rs2 = {rs1 is rs2}")

# Test 2: Add a run
test_run_id = "test-singleton-run"
print(f"\nAdding run: {test_run_id}")
rs1.add_run(test_run_id)

# Test 3: Verify it persists
all_runs = rs2.get_all()
run_ids = [r.run_id for r in all_runs]
print(f"\nAll runs: {len(all_runs)}")
print(f"Test run in list: {test_run_id in run_ids}")

# Test 4: Check file
import json
from pathlib import Path

run_store_path = Path.home() / ".metta" / "run_store.json"
if run_store_path.exists():
    with open(run_store_path) as f:
        data = json.load(f)
    print(f"\nRuns in JSON file: {len(data['runs'])}")
    print(f"Test run in JSON: {test_run_id in data['runs']}")