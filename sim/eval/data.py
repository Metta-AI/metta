import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from omegaconf import DictConfig
import hydra
from rl.eval.eval_stats_db import EvalStatsDB

logger = logging.getLogger(__name__)

def display_name(policy: str, policy_names: Dict[str, str]) -> str:
    return policy_names.get(policy, policy)

def shorten_path(eval_name: str) -> str:
    return eval_name.split('/')[-1]

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

def load_data(cfg: DictConfig, wandb_run) -> Dict[str, pd.DataFrame]:
    logger.info(f"Loading data from {cfg.eval_db_uri}")
    
    try:
        logger.info(f"Connecting to database at {cfg.eval_db_uri}")
        eval_stats_db = EvalStatsDB.from_uri(cfg.eval_db_uri, cfg.run_dir, wandb_run)
        logger.info("Connected to database; Fetching metrics")
        analyzer = hydra.utils.instantiate(cfg.analyzer, eval_stats_db)
        dfs, _ = analyzer.analyze(include_policy_fitness=False)
        metrics_data = construct_metric_to_df_map(cfg, dfs)

        logger.info(f"Retrieved {len(dfs)} metrics from database")
        return metrics_data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}
    
    

def create_policy_hierarchy(policy_names: Set[str]) -> Dict[str, List[str]]:
    """
    Create a hierarchical structure for policies based on prefix patterns.
    
    Args:
        policy_names: Set of policy names
        
    Returns:
        Dictionary mapping policy categories to lists of policy names
    """
    hierarchy = defaultdict(list)
    
    for policy in policy_names:
        # Extract category from policy name based on common prefixes
        if policy.startswith('b.daphne'):
            category = 'daphne'
        elif policy.startswith('navigation_'):
            category = 'navigation'
        elif policy.startswith('b.daveey') or policy.startswith('daveey'):
            category = 'daveey'
        else:
            category = 'other'
        
        hierarchy[category].append(policy)
    
    # Sort policies within each category for consistent ordering
    for category in hierarchy:
        hierarchy[category].sort()
    
    return dict(hierarchy)

def create_eval_hierarchy(eval_names: Set[str]) -> Dict[str, List[str]]:
    """
    Create a hierarchical structure for evaluations based on path components.
    
    Args:
        eval_names: Set of evaluation names
        
    Returns:
        Dictionary mapping evaluation categories to lists of evaluation names
    """
    hierarchy = defaultdict(list)
    
    for eval_name in eval_names:
        # Split by path separator and use first component as category
        components = eval_name.split('/')
        if len(components) > 1:
            category = components[0]
        else:
            category = 'uncategorized'
        
        hierarchy[category].append(eval_name)
    
    # Sort evaluations within each category for consistent ordering
    for category in hierarchy:
        hierarchy[category].sort()
    
    return dict(hierarchy)




def create_policy_hierarchy(policy_names: Set[str]) -> Dict[str, List[str]]:
    """
    Create a hierarchical structure for policies based on prefix patterns.
    
    Args:
        policy_names: Set of policy names
        
    Returns:
        Dictionary mapping policy categories to lists of policy names
    """
    hierarchy = defaultdict(list)
    
    for policy in policy_names:
        # Extract category from policy name based on common prefixes
        if policy.startswith('b.daphne'):
            category = 'daphne'
        elif policy.startswith('navigation_'):
            category = 'navigation'
        elif policy.startswith('b.daveey') or policy.startswith('daveey'):
            category = 'daveey'
        else:
            category = 'other'
        
        hierarchy[category].append(policy)
    
    # Sort policies within each category for consistent ordering
    for category in hierarchy:
        hierarchy[category].sort()
    
    return dict(hierarchy)

def create_eval_hierarchy(eval_names: Set[str]) -> Dict[str, List[str]]:
    """
    Create a hierarchical structure for evaluations based on path components.
    
    Args:
        eval_names: Set of evaluation names
        
    Returns:
        Dictionary mapping evaluation categories to lists of evaluation names
    """
    hierarchy = defaultdict(list)
    
    for eval_name in eval_names:
        # Split by path separator and use first component as category
        components = eval_name.split('/')
        if len(components) > 1:
            category = components[0]
        else:
            category = 'uncategorized'
        
        hierarchy[category].append(eval_name)
    
    # Sort evaluations within each category for consistent ordering
    for category in hierarchy:
        hierarchy[category].sort()
    
    return dict(hierarchy)


def create_matrix_data(
    df: pd.DataFrame, 
    policy_display_names: Dict[str, str],
    eval_display_names: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create data for the policy-evaluation matrix visualization.
    
    Args:
        df: DataFrame containing evaluation metrics
        policy_display_names: Mapping from policy IDs to display names
        eval_display_names: Mapping from evaluation IDs to display names
        
    Returns:
        Dictionary containing matrix visualization data with overall score column
    """
    if df is None or df.empty:
        return {
            'z': [],
            'x': [],
            'y': [],
            'policy_ids': [],
            'eval_ids': []
        }
    
    # Extract unique policies and evaluations in the data
    policies = df['policy_name'].unique()
    evals = df['eval_name'].unique()
    
    # Create a 2D matrix of scores
    mean_col = [col for col in df.columns if col.startswith('mean_')][0]
    
    # Initialize matrix with NaN values
    matrix = np.full((len(policies), len(evals)), np.nan)
    
    # Fill in known values
    for i, policy in enumerate(policies):
        for j, eval_name in enumerate(evals):
            mask = (df['policy_name'] == policy) & (df['eval_name'] == eval_name)
            if mask.any():
                matrix[i, j] = df.loc[mask, mean_col].values[0]
    
    # Calculate overall score for each policy (mean across evals)
    overall_scores = np.nanmean(matrix, axis=1)
    
    # Sort policies by overall score (descending)
    sort_indices = np.argsort(-overall_scores)  # Negative for descending order
    
    # Reorder data by sorted indices
    sorted_policies = [policies[i] for i in sort_indices]
    sorted_policy_labels = [policy_display_names.get(p, p) for p in sorted_policies]
    sorted_matrix = matrix[sort_indices].tolist()
    sorted_overall = overall_scores[sort_indices].tolist()
    
    # Add "Overall" as first evaluation
    eval_labels = ["Overall"] + [eval_display_names.get(e, e) for e in evals]
    eval_ids = ["overall"] + list(evals)
    
    # Add overall scores as first column for each policy
    sorted_matrix_with_overall = []
    for i, row in enumerate(sorted_matrix):
        sorted_matrix_with_overall.append([sorted_overall[i]] + row)
    
    return {
        'z': sorted_matrix_with_overall,
        'x': eval_labels,
        'y': sorted_policy_labels,
        'policy_ids': sorted_policies,
        'eval_ids': eval_ids
    }

def process_data(metric_data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw metric data into formats needed for visualizations.
    
    Args:
        metric_data: Dictionary mapping metric names to dataframes
        config: Application configuration
        
    Returns:
        Dictionary containing processed data for visualizations
    """
    logger.info("Processing data for visualization")
    
    # Initialize processed data structure
    processed = {
        'metric_to_df': metric_data,
        'metrics': list(metric_data.keys()),
    }
    
    # Extract unique evaluation and policy names across all metrics
    all_policy_names = set()
    all_eval_names = set()
    
    for metric, df in metric_data.items():
        if df is None or df.empty:
            continue
            
        if 'policy_name' in df.columns:
            all_policy_names.update(df['policy_name'].unique())
            
        if 'eval_name' in df.columns:
            all_eval_names.update(df['eval_name'].unique())
    
    # Create display name mappings
    policy_display_names = {
        policy: display_name(policy, config.get('policy_names', {}))
        for policy in all_policy_names
    }
    
    eval_display_names = {
        eval_name: shorten_path(eval_name)
        for eval_name in all_eval_names
    }
    
    processed['policy_display_names'] = policy_display_names
    processed['eval_display_names'] = eval_display_names
    
    # Process data for matrix visualization
    logger.info("Creating matrix visualizations")
    for metric in processed['metrics']:
        if metric_data[metric] is not None and not metric_data[metric].empty:
            processed[f'{metric}_matrix'] = create_matrix_data(
                metric_data[metric], 
                processed['policy_display_names'],
                processed['eval_display_names']
            )
    
    return processed