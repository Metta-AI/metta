"""
Data loading and processing for the Metta Policy Evaluation Dashboard.

This module implements the data layer according to the technical design document,
including database access, data transformation, and hierarchical organization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from collections import defaultdict
from rl.eval.eval_stats_db import EvalStatsDB

logger = logging.getLogger(__name__)

def display_name(policy: str, policy_names: Dict[str, str]) -> str:
    return policy_names.get(policy, policy)

def shorten_path(eval_name: str) -> str:
    return eval_name.split('/')[0]

def load_data(eval_db_uri: str, wandb_run, run_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load evaluation data from database.
    
    Args:
        eval_db_uri: URI for evaluation database
        run_dir: Directory containing run data
        
    Returns:
        Dictionary of metric names to pandas DataFrames
    """
    logger.info(f"Loading data from {eval_db_uri}")
    
    try:
        # Initialize database connection
        logger.info(f"Connecting to database at {eval_db_uri}")
        eval_stats_db = EvalStatsDB.from_uri(eval_db_uri, run_dir, wandb_run)
        
        # Get all metrics
        logger.info("Fetching metrics from database")
        metrics_data = eval_stats_db.get_all_metrics()
        logger.info(f"Retrieved {len(metrics_data)} metrics from database")
        
        # Validate and normalize data
        validated_data = {}
        for metric_name, df in metrics_data.items():
            if df is None or df.empty:
                logger.warning(f"Empty data for metric: {metric_name}")
                continue
            
            # Log the dataframe shape and columns
            logger.info(f"Metric {metric_name}: shape={df.shape}, columns={df.columns.tolist()}")
                
            # Ensure required columns exist
            required_cols = ["policy_name", "eval_name"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns {missing_cols} for metric: {metric_name}")
                continue
                
            # Check for mean column
            mean_cols = [col for col in df.columns if col.startswith("mean_")]
            if not mean_cols:
                logger.warning(f"No mean column found for metric: {metric_name}")
                continue
                
            # Add to validated data
            validated_data[metric_name] = df
            logger.info(f"Added validated data for metric: {metric_name}")
        
        if not validated_data:
            logger.warning("No valid metrics found in data")
            return {}
         
        logger.info(f"Successfully loaded {len(validated_data)} metrics from database")
        return validated_data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty dictionary in case of error
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
        Dictionary containing matrix visualization data
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
    
    # Create display name lists
    policy_labels = [policy_display_names.get(p, p) for p in policies]
    eval_labels = [eval_display_names.get(e, e) for e in evals]
    
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
    
    return {
        'z': matrix.tolist(),
        'x': eval_labels,
        'y': policy_labels,
        'policy_ids': policies.tolist(),
        'eval_ids': evals.tolist()
    }

def calculate_pass_rates(
    df: pd.DataFrame,
    threshold: float, 
    policy_display_names: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate pass rates for each policy.
    
    Args:
        df: DataFrame containing episode rewards
        threshold: Threshold score for "passing" an evaluation
        policy_display_names: Mapping from policy IDs to display names
        
    Returns:
        List of dictionaries containing pass rate data
    """
    if df is None or df.empty:
        return []
    
    # Find the mean column
    mean_cols = [col for col in df.columns if col.startswith('mean_')]
    if not mean_cols:
        logger.warning("No mean column found for pass rate calculation")
        return []
        
    mean_col = mean_cols[0]
    
    # Get unique policies and evaluations
    unique_policies = df['policy_name'].unique()
    unique_evals = df['eval_name'].unique()
    
    # Track pass/fail data for each policy
    policy_results = {policy: {"passed": 0, "total": 0} for policy in unique_policies}
    
    # For each policy, check how many evals it passes
    for policy in unique_policies:
        policy_data = df[df['policy_name'] == policy]
        for eval_name in unique_evals:
            eval_policy_data = policy_data[policy_data['eval_name'] == eval_name]
            
            if not eval_policy_data.empty:
                policy_results[policy]["total"] += 1
                if eval_policy_data[mean_col].values[0] >= threshold:
                    policy_results[policy]["passed"] += 1
    
    # Calculate pass rates
    pass_rates = []
    for policy, results in policy_results.items():
        if results["total"] == 0:
            continue
        pass_rate = (results["passed"] / results["total"]) * 100
        display_name = policy_display_names.get(policy, policy)
        
        pass_rates.append({
            "policy_id": policy,
            "display_name": display_name,
            "pass_rate": pass_rate,
            "passed": results["passed"],
            "total": results["total"]
        })
    
    # Sort by pass rate (descending)
    pass_rates.sort(key=lambda x: x["pass_rate"], reverse=True)
    
    return pass_rates

def calculate_policy_ranking(
    df: pd.DataFrame,
    policy_display_names: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate how many evaluations each policy wins.
    
    Args:
        df: DataFrame containing episode rewards
        policy_display_names: Mapping from policy IDs to display names
        
    Returns:
        List of dictionaries containing policy ranking data
    """
    if df is None or df.empty:
        return []
    
    # Find the mean column
    mean_cols = [col for col in df.columns if col.startswith('mean_')]
    if not mean_cols:
        logger.warning("No mean column found for policy ranking calculation")
        return []
        
    mean_col = mean_cols[0]
    
    # Get unique evaluations and policies
    unique_evals = df['eval_name'].unique()
    unique_policies = df['policy_name'].unique()
    
    # Track wins for each policy
    policy_wins = {policy: 0 for policy in unique_policies}
    
    # For each evaluation, find the policy with the highest score
    for eval_name in unique_evals:
        eval_data = df[df['eval_name'] == eval_name]
        if not eval_data.empty:
            max_idx = eval_data[mean_col].idxmax()
            winning_policy = eval_data.loc[max_idx, 'policy_name']
            policy_wins[winning_policy] += 1
    
    # Format results
    ranking = []
    for policy, wins in policy_wins.items():
        display_name = policy_display_names.get(policy, policy)
        ranking.append({
            "policy_id": policy,
            "display_name": display_name,
            "wins": wins
        })
    
    # Sort by number of wins (descending)
    ranking.sort(key=lambda x: x["wins"], reverse=True)
    
    return ranking

def calculate_highest_scores(
    df: pd.DataFrame,
    policy_display_names: Dict[str, str],
    eval_display_names: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate the highest score for each evaluation.
    
    Args:
        df: DataFrame containing episode rewards
        policy_display_names: Mapping from policy IDs to display names
        eval_display_names: Mapping from evaluation IDs to display names
        
    Returns:
        List of dictionaries containing highest score data
    """
    if df is None or df.empty:
        return []
    
    # Find the mean column
    mean_cols = [col for col in df.columns if col.startswith('mean_')]
    if not mean_cols:
        logger.warning("No mean column found for highest scores calculation")
        return []
        
    mean_col = mean_cols[0]
    
    # Get unique evaluations
    unique_evals = df['eval_name'].unique()
    
    # For each evaluation, find the policy with the highest score
    highest_scores = []
    for eval_name in unique_evals:
        eval_data = df[df['eval_name'] == eval_name]
        if not eval_data.empty:
            max_idx = eval_data[mean_col].idxmax()
            winning_policy = eval_data.loc[max_idx, 'policy_name']
            highest_score = eval_data.loc[max_idx, mean_col]
            
            display_eval = eval_display_names.get(eval_name, eval_name)
            display_policy = policy_display_names.get(winning_policy, winning_policy)
            
            highest_scores.append({
                "eval_id": eval_name,
                "display_eval": display_eval,
                "policy_id": winning_policy,
                "display_policy": display_policy,
                "score": highest_score
            })
    
    # Sort by evaluation name
    highest_scores.sort(key=lambda x: x["display_eval"])
    
    return highest_scores

def extract_unique_items(metric_data: Dict[str, pd.DataFrame]) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique policy and evaluation names from metric data.
    
    Args:
        metric_data: Dictionary mapping metric names to dataframes
        
    Returns:
        Tuple of (all_policy_names, all_eval_names)
    """
    all_policy_names = set()
    all_eval_names = set()
    
    for metric, df in metric_data.items():
        if df is None or df.empty:
            continue
            
        if 'policy_name' in df.columns:
            all_policy_names.update(df['policy_name'].unique())
            
        if 'eval_name' in df.columns:
            all_eval_names.update(df['eval_name'].unique())
    
    return all_policy_names, all_eval_names

def create_display_name_mappings(
    all_policy_names: Set[str],
    all_eval_names: Set[str],
    policy_names_config: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Create mappings from internal IDs to display names.
    
    Args:
        all_policy_names: Set of all policy names
        all_eval_names: Set of all evaluation names
        policy_names_config: Configuration mapping policy IDs to display names
        
    Returns:
        Tuple of (policy_display_names, eval_display_names)
    """
    # Create policy display name mapping
    policy_display_names = {
        policy: display_name(policy, policy_names_config)
        for policy in all_policy_names
    }
    
    # Create evaluation display name mapping
    eval_display_names = {
        eval_name: shorten_path(eval_name)
        for eval_name in all_eval_names
    }
    
    return policy_display_names, eval_display_names

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
        'policy_names': config.get('policy_names', {}),
    }
    
    # Extract unique evaluation and policy names across all metrics
    all_policy_names, all_eval_names = extract_unique_items(metric_data)
    
    processed['all_policy_names'] = sorted(list(all_policy_names))
    processed['all_eval_names'] = sorted(list(all_eval_names))
    
    # Create display name mappings
    policy_display_names, eval_display_names = create_display_name_mappings(
        all_policy_names,
        all_eval_names,
        config.get('policy_names', {})
    )
    
    processed['policy_display_names'] = policy_display_names
    processed['eval_display_names'] = eval_display_names
    
    # Create hierarchical structure for policies based on prefix patterns
    processed['policy_hierarchy'] = create_policy_hierarchy(all_policy_names)
    
    # Create hierarchical structure for evaluations based on path components
    processed['eval_hierarchy'] = create_eval_hierarchy(all_eval_names)
    
    # Process data for matrix visualization
    logger.info("Creating matrix visualizations")
    for metric in processed['metrics']:
        if metric_data[metric] is not None and not metric_data[metric].empty:
            processed[f'{metric}_matrix'] = create_matrix_data(
                metric_data[metric], 
                processed['policy_display_names'],
                processed['eval_display_names']
            )
    
    # Process data for performance metrics
    episode_reward_df = next((metric_data[m] for m in metric_data if 'episode_reward' in m), None)
    if episode_reward_df is not None:
        logger.info("Calculating performance metrics")
        
        # Calculate pass rates
        processed['pass_rates'] = calculate_pass_rates(
            episode_reward_df, 
            config.get('pass_threshold', 2.95),
            processed['policy_display_names']
        )
        
        # Calculate policy ranking
        processed['policy_ranking'] = calculate_policy_ranking(
            episode_reward_df,
            processed['policy_display_names']
        )
        
        # Calculate highest scores per evaluation
        processed['highest_scores'] = calculate_highest_scores(
            episode_reward_df,
            processed['policy_display_names'],
            processed['eval_display_names']
        )
    
    return processed