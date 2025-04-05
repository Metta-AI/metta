"""
Functions for loading data from evaluation database.
Adapted from the original code to work without wandb dependency.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import relevant database access code - modify as needed for your environment
# This is a placeholder - you'll need to adapt to your actual data loading logic
try:
    from rl.eval.eval_stats_db import EvalStatsDB
except ImportError:
    logger.warning("Could not import EvalStatsDB. Using mock data loader.")
    
    # Mock implementation for development
    class EvalStatsDB:
        @classmethod
        def from_uri(cls, uri, run_dir, *args):
            return cls()
            
        def get_all_metrics(self):
            # Return mock data structure similar to your real data
            return {
                "episode_reward": pd.DataFrame({
                    "policy_name": ["policy1", "policy1", "policy2", "policy2"],
                    "eval_name": ["eval1", "eval2", "eval1", "eval2"],
                    "mean_episode_reward": [2.5, 1.8, 2.9, 2.7],
                    "std_episode_reward": [0.2, 0.3, 0.1, 0.2]
                })
            }

def load_data(eval_db_uri: str, run_dir: str) -> Dict[str, pd.DataFrame]:
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
        eval_stats_db = EvalStatsDB.from_uri(eval_db_uri, run_dir, None)
        
        # Get all metrics
        metrics_data = eval_stats_db.get_all_metrics()
        
        return metrics_data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Return empty dictionary in case of error
        return {}
    

def process_data(metric_data: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw metric data into formats needed for visualizations.
    
    Args:
        metric_data: Dictionary mapping metric names to dataframes
        config: Application configuration
        
    Returns:
        Dictionary containing processed data for visualizations
    """
    processed = {
        'metric_to_df': metric_data,
        'metrics': list(metric_data.keys()),
        'policy_names': config['policy_names'],
    }
    
    # Extract unique evaluation and policy names across all metrics
    all_eval_names = set()
    all_policy_names = set()
    
    for metric, df in metric_data.items():
        if df is not None and not df.empty:
            if 'eval_name' in df.columns:
                all_eval_names.update(df['eval_name'].unique())
            if 'policy_name' in df.columns:
                all_policy_names.update(df['policy_name'].unique())
    
    processed['all_eval_names'] = sorted(list(all_eval_names))
    processed['all_policy_names'] = sorted(list(all_policy_names))
    
    # Create display name mappings
    processed['policy_display_names'] = {
        policy: get_display_name(policy, config['policy_names'])
        for policy in all_policy_names
    }
    
    processed['eval_display_names'] = {
        eval_name: get_short_eval_name(eval_name)
        for eval_name in all_eval_names
    }
    
    # Create hierarchical structure for policies based on prefix patterns
    processed['policy_hierarchy'] = create_policy_hierarchy(all_policy_names)
    
    # Create hierarchical structure for evaluations based on path components
    processed['eval_hierarchy'] = create_eval_hierarchy(all_eval_names)
    
    # Process data for matrix visualization
    if 'episode_reward' in metric_data:
        processed['matrix_data'] = create_matrix_data(
            metric_data['episode_reward'], 
            processed['policy_display_names'],
            processed['eval_display_names']
        )
    
    # Process data for pass rates
    episode_reward_df = metric_data.get('episode_reward')
    if episode_reward_df is not None:
        processed['pass_rates'] = calculate_pass_rates(
            episode_reward_df, 
            config['pass_threshold'],
            processed['policy_display_names']
        )
    
    return processed

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
    
    # Sort policies within each category
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
    
    # Sort evaluations within each category
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
        df: DataFrame containing episode rewards
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
    policy_labels = [policy_display_names[p] for p in policies]
    eval_labels = [eval_display_names[e] for e in evals]
    
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
    
    # Find the mean column (should be the 3rd column)
    mean_col = [col for col in df.columns if col.startswith('mean_')][0]
    
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