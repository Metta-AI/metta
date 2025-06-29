import datetime as dt
import importlib
import os
import time


def test_queue_latency_helper():
    # Craft a task‑id with the current moment (UTC)
    now = dt.datetime.utcnow()
    ts_str = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    os.environ["SKYPILOT_TASK_ID"] = f"sky-{ts_str}_demo_1"

    mod = importlib.import_module("metta.common.util.skypilot_latency")
    first = mod.queue_latency_s()
    assert first is not None and 0 <= first < 1

    time.sleep(0.02)
    second = mod.queue_latency_s()
    assert second is not None and second >= first
    assert second < 2
