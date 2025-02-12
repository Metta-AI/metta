# Delete this file?
import json
import os
from typing import Any, Dict, List

import numpy as np
from scipy.stats import mannwhitneyu
from tabulate import tabulate
from termcolor import colored


class StatisticalTest:
    def __init__(self, data, categories: List[str]):
        self.categories = categories
        self.prior_scores = {}
        self.stats = {}

        self.policy_names = []
        for episode in data:
            for agent in episode:
                policy_name = agent.get('policy_name', "unknown")
                if policy_name and policy_name not in self.policy_names:
                    self.policy_names.append(policy_name)

        # Initialize stats dictionaries for each stat and policy with None values
        for stat_name in self.categories:
            self.stats[stat_name] = { policy_name: [None] * len(data) for policy_name in self.policy_names }

        # Extract stats per policy per episode
        for idx, episode in enumerate(data):
            # Keep track of which policies participated in this episode
            policies_in_episode = set()
            for agent in episode:
                policy = agent.get('policy_name', "unknown")
                if policy is None:
                    continue
                policies_in_episode.add(policy)
                # Loop through each stat and set this policy's stat for the episode
                for stat_name in self.categories:
                    stat_value = agent.get(stat_name, 0)
                    if self.stats[stat_name][policy][idx] is None:
                        self.stats[stat_name][policy][idx] = stat_value
                    else:
                        self.stats[stat_name][policy][idx] += stat_value

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError

    def format_results(self, results: Dict[str, Any]):
        raise NotImplementedError

    def withHistoricalData(self, prior_scores: Dict[str, Any]):
        self.prior_scores = prior_scores
        return self

class MannWhitneyUTest(StatisticalTest):
    def evaluate(self) -> Dict[str, Any]:
        results = {}
        for stat_name in self.categories:
            if stat_name == 'policy_name':
                continue

            episode_scores = self.stats[stat_name]

            # Collect scores per policy with episode indices
            scores_per_policy = {}
            for policy in self.policy_names:
                scores = []
                for idx, score in enumerate(episode_scores[policy]):
                    if score is not None:
                        scores.append((idx, score))
                scores_per_policy[policy] = scores

            # Collect policy stats (mean, std_dev)
            policy_stats = {}
            for policy in self.policy_names:
                scores = [score for idx, score in scores_per_policy[policy]]
                if scores:
                    if all(isinstance(score, (int, float)) for score in scores):
                        mean = np.mean(scores)
                        std_dev = np.std(scores, ddof=1)
                    else:
                        mean = None
                        std_dev = None
                    policy_stats[policy] = {
                        'mean': mean,
                        'std_dev': std_dev
                    }
                else:
                    policy_stats[policy] = {
                        'mean': None,
                        'std_dev': None
                    }

            # Prepare comparisons
            policy1 = self.policy_names[0]
            comparisons = []
            interpretations = []
            for policy in self.policy_names[1:]:
                # Find common episodes where both policies participated
                episodes_policy1 = set(idx for idx, _ in scores_per_policy[policy1])
                episodes_policy2 = set(idx for idx, _ in scores_per_policy[policy])
                common_episodes = episodes_policy1 & episodes_policy2

                if not common_episodes:
                    # No common episodes; cannot compare
                    interpretations.append("no data")
                    comparison = {
                        'policy1': policy1,
                        'policy2': policy,
                        'u_statistic': None,
                        'p_value': None,
                        'effect_size': None,
                        'interpretation': "no data"
                    }
                    comparisons.append(comparison)
                    continue

                # Extract scores for common episodes
                policy1_scores = [score for idx, score in scores_per_policy[policy1] if idx in common_episodes]
                policy2_scores = [score for idx, score in scores_per_policy[policy] if idx in common_episodes]

                # Perform Mann-Whitney U test
                u_statistic, p_value = mannwhitneyu(policy2_scores, policy1_scores, alternative='two-sided')
                n1 = len(policy1_scores)
                n2 = len(policy2_scores)
                r = 1 - (2 * u_statistic) / (n1 * n2)

                if r > 0 and p_value < 0.05:
                    interpretation = "pos. effect"
                elif r < 0 and p_value < 0.05:
                    interpretation = "neg. effect"
                else:
                    interpretation = "no effect"

                comparisons.append({
                    'policy1': policy1,
                    'policy2': policy,
                    'u_statistic': u_statistic,
                    'p_value': p_value,
                    'effect_size': r,
                    'interpretation': interpretation
                })
                interpretations.append(interpretation)

            summary_value = significance_and_effect(interpretations)

            results[stat_name] = {
                'policy_stats': policy_stats,
                'comparisons': comparisons,
                'summary_value': summary_value
            }
        return results

    def format_results(self, results: Dict[str, Any]):
        data_rows = []
        for stat_name, result in results.items():
            policy_stats = result['policy_stats']
            comparisons = result['comparisons']
            summary_value = result['summary_value']

            # Top row with means and standard deviations
            top_data_row = [stat_name]
            for policy_name in self.policy_names:
                policy_stat = policy_stats[policy_name]
                mean = policy_stat['mean']
                std_dev = policy_stat['std_dev']
                if mean is not None and std_dev is not None:
                    top_data_row.append(f"{mean:.1f} ± {std_dev:.1f}")
                else:
                    top_data_row.append("None ± None")

            # Second row with p-values and effect sizes
            lower_data_row = ['', summary_value]
            for comparison in comparisons:
                p_value = comparison['p_value']
                effect_size = comparison['effect_size']
                interpretation = comparison['interpretation']

                # Apply coloring based on interpretation
                if interpretation == "pos. effect":
                    p_value_str = colored(f"{p_value:.2f}", 'green')
                    effect_size_str = colored(f"{effect_size:.2f}", 'green')
                elif interpretation == "neg. effect":
                    p_value_str = colored(f"{p_value:.2f}", 'red')
                    effect_size_str = colored(f"{effect_size:.2f}", 'red')
                elif interpretation == "no effect":
                    p_value_str = f"{p_value:.2f}" if p_value is not None else "N/A"
                    effect_size_str = f"{effect_size:.2f}" if effect_size is not None else "N/A"
                else:  # "no data"
                    p_value_str = "N/A"
                    effect_size_str = "N/A"

                lower_data_row.append(f"{p_value_str}, {effect_size_str}")

            data_rows.append(top_data_row)
            data_rows.append(lower_data_row)

        # Headers
        headers = ['']
        for policy_name in self.policy_names:
            header = f"{policy_name}\n(mean ± std)\n(p-val, effect size)"
            headers.append(header)

        # Generate the formatted table
        return tabulate(data_rows, headers=headers, tablefmt="fancy_grid")

def significance_and_effect(interpretations):
    all_positive = all(interpretation == "pos. effect" for interpretation in interpretations)
    all_negative = all(interpretation == "neg. effect" for interpretation in interpretations)
    any_positive = any(interpretation == "pos. effect" for interpretation in interpretations)
    any_negative = any(interpretation == "neg. effect" for interpretation in interpretations)
    all_no_effect = all(interpretation == "no effect" for interpretation in interpretations)

    if all_positive:
        return colored("Greater than all!", "green")  # All positive effects
    elif all_negative:
        return colored("Less than all!", "red", attrs=["bold"])  # All negative effects
    elif any_positive and any_negative:
        return colored("Mixed results.", "yellow")  # Some positive, some negative
    elif any_positive and not any_negative:
        return colored("Some pos., none neg.", "yellow")  # Some positive, no negative
    elif any_negative and not any_positive:
        return colored("Some neg., none pos.", "yellow")  # Some negative, no positive
    elif all_no_effect:
        return colored("No significance.", "yellow")  # All no effect
    else:
        return "No interpretation."

# helper function to update historical scores for Elo and Glicko2 tests
def update_scores(historical_scores, new_scores):
    for policy, scores in new_scores.items():
        if policy in historical_scores:
            historical_scores[policy].update(scores)
        else:
            historical_scores[policy] = scores
    return historical_scores

def get_test_results(test: StatisticalTest, scores_path: str = None):
    # SINGLE CHECK FOR EMPTY POLICY LIST:
    if not test.policy_names:
        print("No policies found. Skipping test altogether.")
        return {}, "No results to format (no policies found)."

    historical_data = {}
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r") as file:
            print(f"Loading historical data from {scores_path}")
            try:
                historical_data = json.load(file)
                test.withHistoricalData(historical_data)
            except json.JSONDecodeError:
                print(f"Failed to load historical data from {scores_path}")

    results = test.evaluate()
    formatted_results = test.format_results(results)

    if scores_path:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as file:
            print(f"Saving updated historical data to {scores_path}")
            historical_data.update(results)
            json.dump(historical_data, file, indent=4)

    return results, formatted_results
