"""
Launch a learning-progress hyperparameter sweep locally.

Usage:
    python scripts/lp_sweep.py
"""

from __future__ import annotations

import itertools
import subprocess

BASE_CMD = [
    "uv",
    "run",
    "./tools/run.py",
    "recipes.experiment.cvc.mission_variant_curriculum.train",
    "base_missions=repair",
    "num_cogs=4",
    "trainer.total_timesteps=1000000",
    "checkpointer.epoch_interval=10",
]

Z_VALUES = [10, 20, 30]
EMA_VALUES = [0.001, 0.005]


def main() -> None:
    for idx, (z, ema) in enumerate(itertools.product(Z_VALUES, EMA_VALUES)):
        run_name = f"lp_sweep_z{z}_ema{ema}_run{idx}"
        cmd = BASE_CMD + [
            f"run={run_name}",
            f"training_env.curriculum.algorithm_config.z_score_amplification={z}",
            f"training_env.curriculum.algorithm_config.ema_timescale={ema}",
        ]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

