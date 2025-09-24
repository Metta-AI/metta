from __future__ import annotations

import threading
import time
from pathlib import Path

from metta.common.util.fs import wait_for_file


def test_wait_for_file_requires_minimum_stability(tmp_path: Path) -> None:
    """File stability must be checked at least once even with tiny durations."""

    target = tmp_path / "artifact.bin"

    def writer() -> None:
        with target.open("wb") as handle:
            handle.write(b"A")
            handle.flush()
            time.sleep(0.3)
            handle.write(b"B")
            handle.flush()

    thread = threading.Thread(target=writer)
    thread.start()

    start = time.monotonic()
    assert wait_for_file(
        target,
        timeout=5.0,
        check_interval=0.1,
        stability_duration=0.05,
    )
    elapsed = time.monotonic() - start

    thread.join()

    # Prior bug returned immediately (<0.01s). Require time >= requested stability window.
    assert elapsed >= 0.05
