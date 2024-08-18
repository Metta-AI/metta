
import argparse
import imp
import itertools
import re
import sys
from cv2 import exp

import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats
import numpy as np
from sympy import N
import os
from rl_framework.sample_factory.sample_factory import evaluate

EVALUATION_ARGS = [
    "--no_render",
]

flags = {
    "env_num_agents": [2, 5, 10],
    "env_width": [10, 20, 30, 40, 50],
    "env_height": [10, 20, 30, 40, 50],
    "env_num_altars": [0, 1, 5],
    "env_num_chargers": [0, 1, 5],
    "env_num_generators": [0, 5, 10, 20],
}

def run_eval(sf_args, eval_config: dict):
    argv = sf_args + EVALUATION_ARGS + [
        f"--{k}={v}" for k, v in eval_config.items()
    ]
    return evaluate(argv)

def run_baseline_eval(df, sf_args, eval_config: dict, baseline: str, num_trials=5):
    print(f"Running baseline evaluation with config: {eval_config} and baseline: {baseline}")
    rewards = []
    bl_rewards = []

    for trial in range(num_trials):
        rewards.append(run_eval(sf_args, eval_config))
        bl_rewards.append(run_eval(sf_args, {**eval_config, "experiment": baseline}))
        t_stat, p_value = stats.ttest_rel(rewards, bl_rewards)
        print(
            "Reward: {:.3f} | Baseline Reward: {:.3f} | p-value: {:.3f}".format(
                np.mean(rewards), np.mean(bl_rewards), p_value
            ))

    df = df._append({**eval_config,
                    "bl_reward_mean": round(np.mean(bl_rewards), 3),
                    "bl_reward_std": round(np.std(bl_rewards), 3),
                    "reward_mean": round(np.mean(rewards), 3),
                    "reward_std": round(np.std(rewards), 3),
                    "reward_delta": round(np.mean(rewards) - np.mean(bl_rewards), 3),
                    "p": round(p_value, 3),
                }, ignore_index=True)
    return df

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--num_trials", default=5, type=int)

    args, sf_args = argp.parse_known_args()
    sf_args.append(f"--train_dir={args.train_dir}")
    sf_args.append(f"--experiment={args.experiment}")

    out_path = f"{args.train_dir}/{args.experiment}/eval/"
    os.makedirs(out_path, exist_ok=True)
    flag_combinations = list(itertools.product(*flags.values()))
    flag_combinations = [dict(zip(flags.keys(), combination)) for combination in flag_combinations]

    args.baseline = args.experiment

    data_frame = pd.DataFrame(columns=list(flags.keys()))
    for eval_config in flag_combinations:
        data_frame = run_baseline_eval(data_frame, sf_args, eval_config, args.baseline, args.num_trials)
        with open(f"{out_path}/results.csv", "w") as f:
            data_frame.to_csv(f, index=False)
        with open(f"{out_path}/results.txt", "w") as f:
            print_table(data_frame, file=f, color=False)
        with open(f"{out_path}/results.col", "w") as f:
            print_table(data_frame, file=f, color=True)
        print_table(data_frame, file=sys.stdout, color=True)

def color_rewards(row):
    """
    Colors rewards based on their value and the p-value.
    """
    if row['p'] < 0.05:
        if row['reward_delta'] < 0:
            color = 'red'
        else:
            color = 'green'
        return f"[{color}]{row['reward_delta']}[/]"
    else:
        return str(row['reward_delta'])

def print_table(df, file=sys.stdout, color=True):
    console = Console(file=file)
    table = Table(show_header=True, header_style="bold magenta")
    for column in df.columns:
        table.add_column(column)

    if color:
        df = df.copy()
        df['reward_delta'] = df.apply(color_rewards, axis=1)

    for i in range(len(df)):
        table.add_row(*df.iloc[i].astype(str).tolist())

    console.print(table)

if __name__ == "__main__":
    main()
