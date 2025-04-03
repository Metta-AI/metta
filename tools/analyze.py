import logging
import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig
from rl.wandb.wandb_context import WandbContext
from util.runtime_configuration import setup_metta_environment
from rl.eval.eval_stats_db import EvalStatsDB
from typing import Dict, Any
from collections.abc import Set

def graph_policy_eval_metrics(
    metric_to_df: Dict[str, pd.DataFrame], 
    wandb_run
):
    """
    Parameters:
    - metric_to_df: Dictionary mapping metric names to their corresponding DataFrames
                   Each DataFrame should have columns: policy_name, eval_name, mean_{metric}, std_{metric}
    - wandb_run: wandb run object to which the visualizations will be logged
                       
    """
    logger = logging.getLogger("graph_policy_eval_metrics")
    # Get all metric names
    metric_names = list(metric_to_df.keys())
    
    # Get all dataframes
    dataframes = [metric_to_df[metric] for metric in metric_names]
    
    # Get unique evaluation environments from all dataframes
    all_eval_names: Set[str] = set()
    for df in dataframes:
        if 'eval_name' in df.columns:
            all_eval_names.update(df['eval_name'].unique())
    
    eval_names = sorted(list(all_eval_names))
    
    # Process each evaluation environment
    for eval_name in eval_names:
        # Get a shorter version of the eval name for titles
        short_eval_name = eval_name.split('/')[-1]
        
        # Process each metric
        for metric in metric_names:
            df = metric_to_df[metric]
            
            # Skip if this dataframe doesn't have this eval_name
            if 'eval_name' not in df.columns or eval_name not in df['eval_name'].values:
                continue
            
            # Filter the dataframe for this evaluation environment
            eval_data = df[df['eval_name'] == eval_name].copy()
            
            if not eval_data.empty:
                # Extract the policy names and metric values
                policy_names = eval_data['policy_name'].tolist()
                # Use 3rd column to avoid any formatting differences on metric name
                mean_values = eval_data.iloc[:, 2].tolist()
                

                data = [[label, val] for label, val in zip(policy_names, mean_values)]                
                chart_title = f"{short_eval_name} - {metric}"
                chart = wandb.plot.bar(
                    wandb.Table(data=data, columns=["Policy", f"Mean {metric}"]),
                    "Policy", 
                    f"Mean {metric}",
                    title=chart_title
                )
                
                # Log the chart to W&B
                wandb_run.log({chart_title: chart})
                
                # Also log the raw data as a table for reference
                table_data = []
                for policy, mean in zip(policy_names, mean_values):
                    table_data.append([policy, mean])
                
                table = wandb.Table(
                    data=table_data, 
                    columns=["Policy", f"Mean {metric}"]
                )
                wandb_run.log({f"{chart_title} (data)": table})
                
                logger.info(f"Created W&B visualization for: {chart_title}")
    


def construct_metric_to_df_map(cfg: DictConfig, dfs: list) -> Dict[str, pd.DataFrame]:

    # Extract metrics from config
    metrics = [m.metric for m in cfg.analyzer.analysis.metrics]
    
    # Ensure we have the same number of metrics and dataframes
    if len(metrics) != len(dfs):
        raise ValueError(f"Mismatch between metrics ({len(metrics)}) and dataframes ({len(dfs)})")
    
    metric_to_df = {}
    for metric, df in zip(metrics, dfs):
        metric_to_df[metric] = df
    
    return metric_to_df

@hydra.main(version_base=None, config_path="../configs", config_name="analyzer")
def main(cfg: DictConfig) -> None:
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)
        analyzer =  hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        dfs, _ = analyzer.analyze()

        metric_to_df = construct_metric_to_df_map(cfg, dfs)
        graph_policy_eval_metrics(metric_to_df, wandb_run)


if __name__ == "__main__":
    main()
