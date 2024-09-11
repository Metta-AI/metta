import os
import hydra
from omegaconf import OmegaConf
from rich import traceback
import signal # Aggressively exit on ctrl+c
from rl.wandb.wandb import init_wandb
from rl.carbs.carb_sweep import run_sweep
import pandas as pd
from tabulate import tabulate

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def print_policy_stats(policy_stats):
    # Create a DataFrame with policies as columns
    df = pd.DataFrame({f"Policy {i+1}": {k: v['sum'] / v['count'] for k, v in policy.items()}
                       for i, policy in enumerate(policy_stats)})

    # Calculate percentage and absolute differences from Policy 1
    if len(policy_stats) > 1:
        base_policy = df['Policy 1']
        for col in df.columns[1:]:
            df[f'{col} abs diff'] = (df[col] - base_policy).round(4)
            df[f'{col} % diff'] = ((df[col] - base_policy) / base_policy * 100).round(2)

    # Prepare data for tabulate
    headers = ['Stat', 'Policy 1']
    for i in range(2, len(policy_stats) + 1):
        headers.extend([f'Policy {i}', 'abs diff', '% diff'])

    table_data = [headers]
    for stat in df.index:
        row = [stat, f"{df.loc[stat, 'Policy 1']:.4f}"]
        for i in range(2, len(policy_stats) + 1):
            row.extend([
                f"{df.loc[stat, f'Policy {i}']:.4f}",
                f"{df.loc[stat, f'Policy {i} abs diff']:.4f}",
                f"{df.loc[stat, f'Policy {i} % diff']:.2f}%"
            ])
        table_data.append(row)

    # Create and print the table
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='right')
    print(table)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    framework = hydra.utils.instantiate(cfg.framework, cfg, _recursive_=False)
    if cfg.wandb.track:
        init_wandb(cfg)

    try:
        if cfg.cmd == "train":
            framework.train()

        if cfg.cmd == "evaluate":
            policy_stats = framework.evaluate()
            print_policy_stats(policy_stats)

        if cfg.cmd == "play":
            framework.evaluate()

        if cfg.cmd == "sweep":
            run_sweep(cfg)

    except KeyboardInterrupt:
        os._exit(0)

if __name__ == "__main__":
    main()
