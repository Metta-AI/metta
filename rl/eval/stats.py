from typing import List
import pandas as pd
import scipy.stats as stats
from termcolor import colored
import pandas as pd
from scipy import stats
from termcolor import colored
from typing import List

def significance_test(metrics_df: pd.DataFrame, metric_name: str) -> List[List[str]]:
    """
    Calculates pairwise significance tests between policies for a given metric,
    comparing only policies belonging to the same eval.

    Assumes metrics_df.columns is a MultiIndex with levels ['eval_name', 'policy_name'].

    Returns:
        A list of lists where each inner list contains:
        [formatted_policy1, formatted_policy2, p_value_str, effect_size_str, interpretation, metric_name]
    """
    comparisons = []

    # Group by 'eval_name'. Each group contains columns corresponding to one eval.
    for eval_name, group in metrics_df.groupby(level='eval_name', axis=1):
        # Get the unique policy names in this eval group.
        # Since the group’s columns still carry the MultiIndex, we extract the second level.
        policy_names = list(group.columns.get_level_values('policy_name'))
        n_policies = len(policy_names)

        for i in range(n_policies):
            for j in range(i + 1, n_policies):
                policy1 = policy_names[i]
                policy2 = policy_names[j]

                # Extract the data for each policy.
                # group.xs will slice out the column with the given policy name.
                # (If the result is a DataFrame with one column, convert it to a Series.)
                values1 = group.xs(policy1, level='policy_name', axis=1)
                values2 = group.xs(policy2, level='policy_name', axis=1)
                if isinstance(values1, pd.DataFrame):
                    values1 = values1.iloc[:, 0]
                if isinstance(values2, pd.DataFrame):
                    values2 = values2.iloc[:, 0]

                # Perform Mann-Whitney U test
                u_statistic, p_value = stats.mannwhitneyu(
                    values1,
                    values2,
                    alternative='two-sided'
                )

                # Calculate effect size (r)
                r = 1 - (2 * u_statistic) / (len(values1) * len(values2))

                # Interpret results
                if r > 0 and p_value < 0.05:
                    interpretation = "pos. effect"
                    p_value_str = colored(f"{p_value:.2f}", 'green')
                    effect_size_str = colored(f"{r:.2f}", 'green')
                elif r < 0 and p_value < 0.05:
                    interpretation = "neg. effect"
                    p_value_str = colored(f"{p_value:.2f}", 'red')
                    effect_size_str = colored(f"{r:.2f}", 'red')
                else:
                    interpretation = "no effect"
                    p_value_str = f"{p_value:.2f}"
                    effect_size_str = f"{r:.2f}"

                # Optionally, format long policy names by truncating them.
                def fmt(policy: str) -> str:
                    return policy if len(policy) <= 25 else policy[:5] + '...' + policy[-20:]

                comparisons.append([
                    f"{eval_name} - {fmt(policy1)}",
                    f"{eval_name} - {fmt(policy2)}",
                    p_value_str,
                    effect_size_str,
                    interpretation,
                    metric_name
                ])

    return comparisons
