import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def safe_diff(a, b):
    a, b = safe_float(a), safe_float(b)
    if np.isnan(a) or np.isnan(b):
        return "-"
    return a - b

def safe_percent_diff(a, b):
    a, b = safe_float(a), safe_float(b)
    if np.isnan(a) or np.isnan(b) or b == 0:
        return "-"
    return (a - b) / b * 100

def safe_stdev_diff(a, b, stdev):
    a, b, stdev = safe_float(a), safe_float(b), safe_float(stdev)
    if np.isnan(a) or np.isnan(b) or np.isnan(stdev) or stdev == 0:
        return "-"
    return (a - b) / stdev

def get_stat_value(stat):
    if isinstance(stat, dict):
        return safe_float(stat.get('sum', np.nan) / stat.get('count', 1))
    return safe_float(stat)

def print_policy_stats(policy_stats):
    # Create a DataFrame with policies as columns
    df = pd.DataFrame({f"Policy {i+1}": {k: get_stat_value(v) for k, v in policy.items()}
                       for i, policy in enumerate(policy_stats)})

    # Sort the DataFrame index (stats) alphabetically
    df = df.sort_index()

    # Calculate differences from Policy 2
    if len(policy_stats) > 1:
        base_policy = df['Policy 2']
        df['Policy 1 abs diff'] = df.apply(lambda row: safe_diff(row['Policy 1'], row['Policy 2']), axis=1)
        df['Policy 1 % diff'] = df.apply(lambda row: safe_percent_diff(row['Policy 1'], row['Policy 2']), axis=1)
        for col in df.columns[2:]:
            if col.startswith('Policy') and col != 'Policy 2':
                df[f'{col} abs diff'] = df.apply(lambda row: safe_diff(row[col], row['Policy 2']), axis=1)
                df[f'{col} % diff'] = df.apply(lambda row: safe_percent_diff(row[col], row['Policy 2']), axis=1)

    # Prepare headers for tabulate
    headers = ['Stat', 'Policy 1', 'abs diff', '% diff']
    for i in range(2, len(policy_stats) + 1):
        headers.extend([f'Policy {i}'])

    # Prepare table data with visual grouping and color coding
    table_data = []
    prev_first_token = None
    for stat in df.index:
        first_token = stat.split('.')[0]
        if first_token != prev_first_token and prev_first_token is not None:
            # Add an empty row to create a thicker border
            table_data.append([''] * len(headers))

        policy1_value = df.loc[stat, 'Policy 1']
        policy1_abs_diff = df.loc[stat, 'Policy 1 abs diff']
        policy1_pct_diff = df.loc[stat, 'Policy 1 % diff']

        # Color coding for Policy 1
        if isinstance(policy1_abs_diff, (int, float)) and policy1_abs_diff > 0:
            policy1_color = 'green'
        elif isinstance(policy1_abs_diff, (int, float)) and policy1_abs_diff < 0:
            policy1_color = 'red'
        else:
            policy1_color = None

        row = [
            stat,
            colored(f"{policy1_value:.4f}" if isinstance(policy1_value, (int, float)) else str(policy1_value), policy1_color),
            colored(f"{'+' if isinstance(policy1_abs_diff, (int, float)) and policy1_abs_diff >= 0 else ''}{policy1_abs_diff:.4f}" if isinstance(policy1_abs_diff, (int, float)) else str(policy1_abs_diff), policy1_color),
            colored(f"{'+' if isinstance(policy1_pct_diff, (int, float)) and policy1_pct_diff >= 0 else ''}{policy1_pct_diff:.2f}%" if isinstance(policy1_pct_diff, (int, float)) else str(policy1_pct_diff), policy1_color)
        ]

        for i in range(2, len(policy_stats) + 1):
            policy_value = df.loc[stat, f'Policy {i}']
            row.append(f"{policy_value:.4f}" if isinstance(policy_value, (int, float)) else str(policy_value))
        table_data.append(row)
        prev_first_token = first_token

    # Create and print the table
    table = tabulate(table_data, headers=headers, tablefmt='grid', numalign='right')
    print(table)
