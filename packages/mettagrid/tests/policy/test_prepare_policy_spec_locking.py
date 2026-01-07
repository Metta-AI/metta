from __future__ import annotations

import json
import multiprocessing
from pathlib import Path


def _load_policy_spec_worker(submission_dir: str) -> None:
    from mettagrid.policy.prepare_policy_spec import load_policy_spec_from_path

    load_policy_spec_from_path(Path(submission_dir))


def test_setup_script_runs_once_across_processes(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()

    (submission_dir / "policy_spec.json").write_text(
        json.dumps(
            {
                "class_path": "foo.Bar",
                "data_path": None,
                "init_kwargs": {},
                "setup_script": "setup.py",
            }
        )
    )
    (submission_dir / "setup.py").write_text(
        "\n".join(
            [
                "import os",
                "import time",
                "",
                "time.sleep(0.2)",
                "with open('setup_runs.txt', 'a', encoding='utf-8') as f:",
                '    f.write(f"{os.getpid()}\\n")',
                "",
            ]
        )
    )

    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_load_policy_spec_worker, args=(str(submission_dir),)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert [process.exitcode for process in processes] == [0, 0, 0, 0]

    runs = (submission_dir / "setup_runs.txt").read_text(encoding="utf-8").splitlines()
    assert len(runs) == 1
