import pandas as pd
from scipy import stats
from termcolor import colored
def calculate_significance_tests(metrics_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Calculates pairwise significance tests between policies for a given metric.
    Uses Mann-Whitney U test (non-parametric) since we can't assume normal distribution.

    Returns a DataFrame with p-values for each policy pair comparison.
    """
    policies = metrics_df.keys()
    n_policies = len(policies)
    comparisons = []

    # Calculate pairwise significance
    for i in range(n_policies):
        for j in range(i + 1, n_policies):
            policy1, policy2 = policies[i], policies[j]
            values1 = metrics_df[policy1]
            values2 = metrics_df[policy2]

            # Perform Mann-Whitney U test
            u_statistic, p_value = stats.mannwhitneyu(
                values1,
                values2,
                alternative='two-sided'
            )
            n1 = len(values1)
            n2 = len(values2)
            r = 1 - (2 * u_statistic) / (n1 * n2)

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
                p_value_str = f"{p_value:.2f}" if p_value is not None else "N/A"
                effect_size_str = f"{r:.2f}" if r is not None else "N/A"

            comparisons.append([
                policy1[:5] + '...' + policy1[-20:] if len(policy1) > 25 else policy1,
                policy2[:5] + '...' + policy2[-20:] if len(policy2) > 25 else policy2,
                p_value_str,
                effect_size_str,
                interpretation,
                metric_name
            ])

    return comparisons
