from scipy.stats import mannwhitneyu
import numpy as np
from tabulate import tabulate
from termcolor import colored

class StatisticalTest:
    def __init__(self, stats, policy_names, categories_list):
        self.stats = stats
        self.policy_names = policy_names
        self.categories_list = categories_list
        self.results = None  # Raw results
        self.formatted_results = None  # Formatted string for display

    def run_test(self):
        raise NotImplementedError

    def get_results(self):
        return self.results

    def format_results(self):
        raise NotImplementedError

    def get_formatted_results(self):
        if self.formatted_results is None:
            self.format_results()
        return self.formatted_results

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

class MannWhitneyUTest(StatisticalTest):
    def run_test(self):
        self.results = {}
        for stat_name in self.categories_list:
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

            self.results[stat_name] = {
                'policy_stats': policy_stats,
                'comparisons': comparisons,
                'summary_value': summary_value
            }

    def format_results(self):
        data_rows = []
        for stat_name, result in self.results.items():
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
        self.formatted_results = tabulate(data_rows, headers=headers, tablefmt="fancy_grid")

class EloTest(StatisticalTest):
    def run_test(self):
        winning_margin = 1  # Set a minimum winning margin for a win
        all_scores = next(iter(self.stats.values()))
        total_episodes = len(next(iter(all_scores.values())))

        elo = {policy: 1000 for policy in self.policy_names}

        def get_elo_constant(e):
            e += 1
            Ki = 32
            Kf = 2
            return Ki + (Kf - Ki) / (total_episodes - 1) * (e - 1)

        for episode_no in range(total_episodes):
            K = get_elo_constant(episode_no)
            # Get policies that participated in this episode
            participating_policies = [policy for policy in self.policy_names if all_scores[policy][episode_no] is not None]

            # If fewer than 2 policies participated, skip this episode
            if len(participating_policies) < 2:
                continue

            for i in range(len(participating_policies)):
                for j in range(i + 1, len(participating_policies)):
                    policy_i = participating_policies[i]
                    policy_j = participating_policies[j]
                    score_i = all_scores[policy_i][episode_no]
                    score_j = all_scores[policy_j][episode_no]
                    if score_i is not None and score_j is not None:
                        E_i = 1 / (1 + 10 ** ((elo[policy_j] - elo[policy_i]) / 400))
                        E_j = 1 - E_i
                        if (score_i - score_j) > winning_margin:
                            S_i = 1
                            S_j = 0
                        elif (score_j - score_i) > winning_margin:
                            S_i = 0
                            S_j = 1
                        else:
                            S_i = 0.5
                            S_j = 0.5
                        elo[policy_i] += K * (S_i - E_i)
                        elo[policy_j] += K * (S_j - E_j)

        self.results = elo

    def format_results(self):
        # Headers
        headers = ['Policy', 'Elo Rating']
        data_rows = []

        max_elo = max(self.results.values())
        min_elo = min(self.results.values())

        for policy in self.policy_names:
            elo_rating = self.results[policy]
            elo_rounded = round(elo_rating)
            if elo_rating == max_elo:
                elo_str = colored(f"{elo_rounded}", "green")
            elif elo_rating == min_elo:
                elo_str = colored(f"{elo_rounded}", "red")
            else:
                elo_str = colored(f"{elo_rounded}", "yellow")
            data_rows.append([policy, elo_str])

        self.formatted_results = tabulate(data_rows, headers=headers, tablefmt="fancy_grid")

# Placeholder classes for KruskalWallisTest and Glicko2Test
class KruskalWallisTest(StatisticalTest):
    def run_test(self):
        # Implement the Kruskal-Wallis test here
        pass

    def format_results(self):
        # Implement formatting logic for Kruskal-Wallis test
        pass

class Glicko2Test(StatisticalTest):
    def run_test(self):
        # Implement the Glicko-2 rating system here
        pass

    def format_results(self):
        # Implement formatting logic for Glicko-2 test
        pass
