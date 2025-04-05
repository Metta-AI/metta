import logging
import hydra
import wandb
from datetime import datetime
import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from omegaconf import DictConfig, OmegaConf
from rl.wandb.wandb_context import WandbContext
from util.runtime_configuration import setup_metta_environment
from rl.eval.eval_stats_db import EvalStatsDB
from typing import Dict, Any, List, Tuple, Optional
from collections.abc import Set

def get_display_name(policy_name: str, policy_names: Optional[Dict[str, str]] = None) -> str:
    """Maps policy's internal name to display name if available"""
    if policy_names is not None and policy_name in policy_names:
        return policy_names[policy_name]
    return policy_name

def get_short_eval_name(eval_name: str) -> str:
    """Extracts the short name from a full evaluation path"""
    return eval_name.split('/')[-1]

def graph_policy_eval_metrics(
    metric_to_df: Dict[str, pd.DataFrame], 
    wandb_run,
    policy_names: Optional[Dict[str, str]] = None
):
    """
    Creates bar charts visualizing metrics for each policy across evaluation environments.
    
    Dataframe structure expectations:
    - metric_to_df maps metric names to dataframes with columns:
      policy_name, eval_name, mean_{metric}, std_{metric}
    - Each row represents one policy-eval combination
    """
    logger = logging.getLogger(__name__)
    metric_names = list(metric_to_df.keys())
    dataframes = [metric_to_df[metric] for metric in metric_names]
    
    all_eval_names: Set[str] = set()
    for df in dataframes:
        if 'eval_name' in df.columns:
            all_eval_names.update(df['eval_name'].unique())
    eval_names = sorted(list(all_eval_names))
    
    for eval_name in eval_names:
        short_eval_name = get_short_eval_name(eval_name)
        
        for metric in metric_names:
            df = metric_to_df[metric]
            
            if 'eval_name' not in df.columns or eval_name not in df['eval_name'].values:
                continue
            eval_data = df[df['eval_name'] == eval_name].copy()            
            if not eval_data.empty:
                policy_names_list = eval_data['policy_name'].tolist()
                mean_values = eval_data.iloc[:, 2].tolist()
                
                display_names = [get_display_name(name, policy_names) for name in policy_names_list]
                display_metric = metric.replace('.', '_')                
                chart_title = f"{short_eval_name} - {display_metric}"
                
                # Create custom bar chart with constrained y-axis
                chart_data = {
                    "data": [
                        {
                            "x": display_names,
                            "y": mean_values,
                            "type": "bar",
                            "name": f"Mean {display_metric}"
                        }
                    ],
                    "layout": {
                        "title": chart_title,
                        "xaxis": {"title": "Policy"},
                        "yaxis": {
                            "title": f"Mean {display_metric}",
                            "range": [0, 3]  # Constrain y-axis to [0, 3]
                        }
                    }
                }
                
                fig = go.Figure(data=chart_data["data"], layout=chart_data["layout"])

                wandb_run.log({chart_title: fig})                
                # Also log the raw data as a table for reference
                table_data = []
                for policy, display_name, mean in zip(policy_names_list, display_names, mean_values):
                    table_data.append([display_name, mean, policy])                
                table = wandb.Table(
                    data=table_data, 
                    columns=["Policy", f"Mean {display_metric}", "ID"]
                )
                wandb_run.log({f"{chart_title} (data)": table})
                
                logger.info(f"Created W&B visualization for: {chart_title}")

def graph_pass_rate(
    df: pd.DataFrame,
    threshold: float,
    wandb_run,
    policy_names: Optional[Dict[str, str]] = None
):
    """
    Creates pass rate graph for each policy across all evals.
    
    For each unique (policy, eval) combination:
    - An evaluation "passes" if mean score >= threshold
    - Pass rate = (passing evals / total evals) * 100
    """
    logger = logging.getLogger(__name__)
    
    if df is None or df.empty:
        logger.warning("No episode_reward data available for pass rate calculation")
        return
    
    unique_policies = df['policy_name'].unique()
    unique_evals = df['eval_name'].unique()
    
    # Track pass/fail data for each policy
    policy_results = {policy: {"passed": 0, "total": 0} for policy in unique_policies}
    
    # Find the mean column (should be the 3rd column)
    mean_col = df.columns[2]  # Assumes mean_{metric} is the 3rd column
    metric_name = df.columns[2][5:]
    
    # For each policy, check how many evals it passes
    for policy in unique_policies:
        policy_data = df[df['policy_name'] == policy]
        for eval_name in unique_evals:
            eval_policy_data = policy_data[policy_data['eval_name'] == eval_name]
            
            if not eval_policy_data.empty:
                policy_results[policy]["total"] += 1
                if eval_policy_data[mean_col].values[0] >= threshold:
                    policy_results[policy]["passed"] += 1
    
    pass_rates = []
    for policy, results in policy_results.items():
        if results["total"] == 0:
            continue
        # Calculate actual percentage, not just the total
        pass_rate = (results["passed"] / results["total"]) * 100
        pass_rates.append([policy, pass_rate, results["passed"], results["total"]])
    pass_rates.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Pass rates: {pass_rates}")

    chart_data = []
    for policy, pass_rate, passing, total in pass_rates:
        display_name = get_display_name(policy, policy_names)
        chart_data.append([display_name, pass_rate, f"{passing}/{total}", policy])
    # Ensure at least one policy is showing
    if not chart_data:
        logger.warning("No pass rate data to display")
        return
        
    # TODO: Change this to plotly so we can set the x-axis max to 100%
    chart_table = wandb.Table(data=chart_data, 
                            columns=["Policy", "Pass Rate (%)", "Passing Evals / Total", "ID"])
    chart = wandb.plot.bar(
        chart_table,
        "Policy",
        "Pass Rate (%)",
        title=f"Policy Pass Rate ({metric_name} >= {threshold})"
    )
    wandb_run.log({"Policy Pass Rate": chart})
    wandb_run.log({"Policy Pass Rate (data)": chart_table})

    logger.info("Created W&B visualization for Policy Pass Rate")

def graph_highest_scores_per_eval(
    df: pd.DataFrame,
    wandb_run,
    policy_names: Optional[Dict[str, str]] = None
):
    """
    Creates a bar chart showing highest score for each eval across policies.
    """
    logger = logging.getLogger(__name__)
    unique_evals = df['eval_name'].unique()
    # Assumes the 3rd column is the mean value column (e.g. "mean_episode_reward")
    mean_col = df.columns[2]
    metric_name = df.columns[2][5:]  # strip "mean_"
    
    highest_scores = []
    for eval_name in unique_evals:
        eval_data = df[df['eval_name'] == eval_name].copy()
        if not eval_data.empty:
            max_idx = eval_data[mean_col].idxmax()
            winning_policy = eval_data.loc[max_idx, 'policy_name']
            highest_score = eval_data.loc[max_idx, mean_col]
            short_eval_name = get_short_eval_name(eval_name)
            display_policy_name = get_display_name(winning_policy, policy_names)
            highest_scores.append([short_eval_name, highest_score, display_policy_name, winning_policy])
        else:
            logger.warning(f"No data for eval {eval_name}")
    
    highest_scores.sort(key=lambda x: x[0])  # Sort by eval name (index 0)
    highest_score_table = wandb.Table(
        data=highest_scores,
        columns=["Eval", "Score", "Winning Policy", "ID"]
    )
    highest_score_chart = wandb.plot.bar(
        highest_score_table,
        "Eval",
        "Score",
        title=f"Highest {metric_name}"
    )
    wandb_run.log({f"Highest {metric_name} Leaderboard": highest_score_chart})
    wandb_run.log({f"Highest {metric_name} Leaderboard (data)": highest_score_table})
    
    logger.info(f"Created W&B visualization for Highest {metric_name}")

def graph_leaderboard(
    df: pd.DataFrame,
    wandb_run,
    policy_names: Optional[Dict[str, str]] = None
):
    """
    Creates a bar chart showing how many evals each policy won based on a metric.
    """
    logger = logging.getLogger(__name__)

    unique_evals = df['eval_name'].unique()
    all_policies = sorted(df['policy_name'].unique())
    # Find the mean column (should be the 3rd column)
    mean_col = df.columns[2]  # Assumes mean_{metric} is the 3rd column
    metric_name = df.columns[2][5:]
    
    policy_wins = {policy: 0 for policy in all_policies}
    
    for eval_name in unique_evals:
        # Filter the dataframe for this evaluation environment
        eval_data = df[df['eval_name'] == eval_name].copy()
        
        if not eval_data.empty:
            # Find the policy with the highest score
            max_idx = eval_data[mean_col].idxmax()
            winning_policy = eval_data.loc[max_idx, 'policy_name']
            policy_wins[winning_policy] += 1
    
    policy_win_data = []
    for policy, wins in policy_wins.items():
        display_name = get_display_name(policy, policy_names)
        policy_win_data.append([display_name, wins, policy])
    policy_win_data.sort(key=lambda x: x[1], reverse=True)
    
    wins_column_name = "Wins"
    policy_win_table = wandb.Table(
        data=policy_win_data,
        columns=["Policy", wins_column_name, "ID"]
    )
    policy_win_chart = wandb.plot.bar(
        policy_win_table,
        "Policy",
        wins_column_name,
        title=f"Policy Leaderboard (# of eval wins)"
    )
    
    wandb_run.log({"Policy Leaderboard": policy_win_chart})
    wandb_run.log({"Policy Leaderboard (data)": policy_win_table})
    
    logger.info("Created W&B visualization for Policy Leaderboard")

def construct_metric_to_df_map(cfg: DictConfig, dfs: list) -> Dict[str, pd.DataFrame]:
    """
    Constructs a mapping from metric names to their respective dataframes.
    
    The analyzer's configuration defines a list of metrics to analyze (in cfg.analyzer.analysis.metrics).
    For each metric, the analyzer produces one dataframe, resulting in a 1:1 correspondence
    between metrics and dataframes. The dataframe will contain scores for all (eval, policy) pairs.
    
    Expected dataframe schema for each metric:
    - policy_name: String identifier for the policy
    - eval_name: String identifier for the evaluation environment
    - mean_{metric}: Float value representing the mean of the metric for this policy in this eval
    - std_{metric}: Float value representing the standard deviation of the metric
    
    Parameters:
    - cfg: Configuration object containing analyzer settings
    - dfs: List of dataframes produced by the analyzer, one per metric
    
    Returns:
    - Dictionary mapping from metric names to their corresponding dataframes
    """
    metrics = [m.metric for m in cfg.analyzer.analysis.metrics]
    
    if len(metrics) != len(dfs):
        raise ValueError(f"Mismatch between metrics ({len(metrics)}) and dataframes ({len(dfs)})")
    
    metric_to_df = {}
    for metric, df in zip(metrics, dfs):
        metric_to_df[metric] = df
    
    return metric_to_df

def get_episode_reward_df(metric_to_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extracts the episode_reward dataframe from the metrics dictionary.
    Returns None if not found.
    """
    # Find any metric that contains "episode_reward" in its name
    episode_reward_metric = None
    for metric_name in metric_to_df.keys():
        if "episode_reward" in metric_name:
            episode_reward_metric = metric_name
            break
    
    if episode_reward_metric is None:
        return None
    
    # Get the episode reward dataframe
    return metric_to_df[episode_reward_metric]

def generate_report(cfg: DictConfig):
    logger = logging.getLogger("generate_report")
    
    # Set up a one-off run on wandb to avoid conflicts on graphs when iterating
    one_off_cfg = copy.deepcopy(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    one_off_cfg.run = f"{cfg.run}.report.{timestamp}"
    one_off_cfg.run_dir = cfg.run_dir
    with WandbContext(one_off_cfg) as wandb_run:

        eval_stats_db = EvalStatsDB.from_uri(cfg.eval.eval_db_uri, cfg.run_dir, wandb_run)
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        dfs, _ = analyzer.analyze(include_policy_fitness=False)

        metric_to_df = construct_metric_to_df_map(cfg, dfs)
        
        # Extract episode_reward dataframe for specific graphs
        episode_reward_df = get_episode_reward_df(metric_to_df)
        if episode_reward_df is None:
            logger.warning("No episode_reward metric found in the data. Some visualizations will be skipped.")
        
        # TODO: Move this into config somewhere
        policy_names = {
            "b.daphne.navigation_varied_obstacle_shapes_pretrained.r.1": "varied_obstacles_pretrained",
            "b.daphne.navigation_varied_obstacle_shapes.r.0": "varied_obstacles",
            "navigation_poisson_sparser.r.2": "3_objects_far",
            "navigation_infinite_cooldown_sparser_pretrained.r.0": "inf_cooldown_sparse_pretrained",
            "navigation_infinite_cooldown_sparser.r.0": "inf_cooldown_sparse",
            "navigation_infinite_cooldown_sweep:v46": "inf_cooldown:v46",
            "navigation_poisson_sparser_pretrained.r.6": "3_objects_far_pretrained",
            "navigation_infinite_cooldown_sweep": "inf_cooldown",
            "navigation_infinite_cooldown_sweep.r.0": "inf_cooldown2",
            "b.daveey.t.8.rdr9.3": "daveey.t.8.rdr9.3",
            "b.daveey.t.4.rdr9.3": "daveey.t.4.rdr9.3",
            "b.daveey.t.8.rdr9.mb2.1": "daveey.t.8.rdr9.mb2.1",
            "daveey.t.1.pi.dpm": "daveey.t.1.pi.dpm",
            "b.daveey.t.64.dr90.1": "daveey.t.64.dr90.1",
            "b.daveey.t.8.rdr9.sb": "daveey.t.8.rdr9.sb",
        }
        
        # Override with config if available
        if 'policy_names' in cfg and OmegaConf.is_dict(cfg.policy_names):
            policy_names = OmegaConf.to_container(cfg.policy_names)
            logger.info(f"Using policy name mapping from config: {policy_names}")
        
        # Generate visualizations
        logger.info("Generating visualizations")
        
        # 1. Generic metric visualizations
        logger.info("Graphing policy eval metrics")
        graph_policy_eval_metrics(metric_to_df, wandb_run, policy_names)
        
        # 2. Episode reward-specific visualizations
        if episode_reward_df is not None:
            logger.info("Graphing pass rate")
            graph_pass_rate(episode_reward_df, 2.95, wandb_run, policy_names)
            
            logger.info("Graphing leaderboard")
            graph_leaderboard(episode_reward_df, wandb_run, policy_names)
            
            logger.info("Graphing highest scores")
            graph_highest_scores_per_eval(episode_reward_df, wandb_run, policy_names)