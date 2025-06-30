import datetime
import importlib
import os
import time
import uuid

import pytest


@pytest.mark.parametrize("prefix", ["sky-", "managed-sky-", "sky-managed-"])
def test_queue_latency_helper(prefix):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
    os.environ["SKYPILOT_TASK_ID"] = f"{prefix}{ts}_demo_{uuid.uuid4().hex[:3]}"
    mod = importlib.import_module("metta.common.util.skypilot_latency")
    t0 = mod.queue_latency_s()
    assert t0 is not None and 0 <= t0 < 1
    time.sleep(0.02)
    t1 = mod.queue_latency_s()
    assert t1 is not None and t1 >= t0 < 2
    del os.environ["SKYPILOT_TASK_ID"]
