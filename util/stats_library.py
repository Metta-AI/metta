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

class EloTest(StatisticalTest):
    def evaluate(self) -> Dict[str, Any]:
        scores = {}
        for policy in self.policy_names:
            scores[policy] = self.prior_scores.get(policy, {'score': 1000, 'matches': 0})

        winning_margin = 1  # Set a minimum winning margin for a win
        # Use the first stat in categories_list instead of hardcoding 'altar'
        stat_name = self.categories[0]
        all_scores = self.stats[stat_name]
        total_episodes = len(all_scores[self.policy_names[0]])



        for episode_no in range(total_episodes):
            # Get policies that participated in this episode
            participating_policies = [
                policy for policy in self.policy_names
                if all_scores[policy][episode_no] is not None
            ]

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
                        # Retrieve current ELO scores
                        elo_i = scores[policy_i]['score']
                        elo_j = scores[policy_j]['score']

                        # Calculate expected scores
                        E_i = 1 / (1 + 10 ** ((elo_j - elo_i) / 400))
                        E_j = 1 - E_i

                        # Determine actual scores
                        if (score_i - score_j) > winning_margin:
                            S_i, S_j = 1, 0
                        elif (score_j - score_i) > winning_margin:
                            S_i, S_j = 0, 1
                        else:
                            S_i, S_j = 0.5, 0.5

                        # Get K values per policy
                        K_i = self._get_k(scores, policy_i)
                        K_j = self._get_k(scores, policy_j)

                        # Update elo ratings
                        scores[policy_i]['score'] += K_i * (S_i - E_i)
                        scores[policy_j]['score'] += K_j * (S_j - E_j)

                        # Update episodes played per policy
                        scores[policy_i]['matches'] += 1
                        scores[policy_j]['matches'] += 1

        return scores

    def format_results(self, results: Dict[str, Any]):
        # Headers
        headers = ['Policy', 'Elo Rating', 'Matches Played']
        data_rows = []

        elo_values = [data['score'] for data in results.values()]
        max_elo = max(elo_values)
        min_elo = min(elo_values)

        for policy in self.policy_names:
            elo_rating = results[policy]['score']
            episodes_played = results[policy]['matches']
            elo_rounded = round(elo_rating)
            if elo_rating == max_elo:
                elo_str = colored(f"{elo_rounded}", "green")
            elif elo_rating == min_elo:
                elo_str = colored(f"{elo_rounded}", "red")
            else:
                elo_str = colored(f"{elo_rounded}", "yellow")
            data_rows.append([policy, elo_str, episodes_played])

        return tabulate(data_rows, headers=headers, tablefmt="fancy_grid")

    def _get_k(self, scores, policy):
        n = scores[policy]['matches']
        # Adjust K based on the number of episodes played by this policy
        if n < 30:
            return 32
        elif n < 100:
            return 16
        else:
            return 8
# Placeholder classes for KruskalWallisTest for multiplayer significance analysis
class KruskalWallisTest(StatisticalTest):
    def evaluate(self) -> Dict[str, Any]:
        # Not implemented yet
        pass

    def format_results(self, results: Dict[str, Any]):
        # Not implemented yet
        pass

class Glicko2Test(StatisticalTest):
    def __init__(self, stats, categories: List[str],
                 tau=1.1, epsilon=0.000001, max_iterations=10):
        super().__init__(stats, categories)

        self.initial_mu = 0.0 # by definition
        self.initial_phi = 2.014761872416068 # φ = RD / 173.7178, RD = 350 ('tis what Mark Glickman used)
        self.initial_sigma = 0.06 # Starting volatility for new players. Can adjust.
        self.tau = tau
        self.epsilon = epsilon
        self.max_iterations = max_iterations


    def evaluate(self, verbose: bool = False) -> Dict[str, Any]:
        ratings = {}

        for policy in self.policy_names:
            ratings[policy] = {
                'mu': self.initial_mu,
                'phi': self.initial_phi,
                'sigma': self.initial_sigma,
                'matches': 0
            }
            # Use initial ratings if provided, else default values
            if policy in self.prior_scores:
                rating_info = self.prior_scores[policy]
                ratings[policy]['mu'] = (rating_info['rating'] - 1500) / 173.7178  # Convert rating to mu
                ratings[policy]['phi'] = rating_info['RD'] / 173.7178  # Convert RD to phi
                ratings[policy]['sigma'] = rating_info.get('sigma', self.initial_sigma)
                ratings[policy]['matches'] = rating_info.get('matches', 0)

        # Collect all match results over all episodes
        stat_name = self.categories[0]
        all_scores = self.stats[stat_name]
        total_episodes = len(all_scores[self.policy_names[0]])

        # For each policy, collect matches over all episodes
        policy_matches = {policy: [] for policy in self.policy_names}
        # Extra metrics for checking if Glicko is doing the job
        verbose_results = {
            policy: {
                'total_score': 0,
                'wins': 0,
                'losses': 0,
                'win_pct': 0,
                'avg_score': 0,
                'rank': 0,
                'glicko2': 0
            } for policy in self.policy_names
        }

        for episode_no in range(total_episodes):
            # Get policies that participated in this episode
            participating_policies = [
                policy for policy in self.policy_names
                if all_scores[policy][episode_no] is not None
            ]

            # Skip episodes with more or less than 2 policies
            if len(participating_policies) != 2:
                continue

            # For each pair of policies, determine outcome
            for i in range(len(participating_policies)):
                for j in range(i + 1, len(participating_policies)):
                    policy_i = participating_policies[i]
                    policy_j = participating_policies[j]
                    score_i = all_scores[policy_i][episode_no]
                    score_j = all_scores[policy_j][episode_no]

                    # Determine outcome
                    if (score_i - score_j) > 1:
                        s_i = 1.0  # policy_i wins
                        s_j = 0.0  # policy_j loses
                    elif (score_j - score_i) > 1:
                        s_i = 0.0  # policy_i loses
                        s_j = 1.0  # policy_j wins
                    else:
                        s_i = 0.5  # Draw
                        s_j = 0.5

                    #delete aux metrics after testing
                    verbose_results[policy_i]['total_score'] += score_i
                    verbose_results[policy_j]['total_score'] += score_j
                    if s_i == 1:
                        verbose_results[policy_i]['wins'] += 1
                        verbose_results[policy_j]['losses'] += 1
                    elif s_j == 1:
                        verbose_results[policy_j]['wins'] += 1
                        verbose_results[policy_i]['losses'] += 1

                    # Record matches
                    policy_matches[policy_i].append({
                        'opponent': policy_j,
                        'score': s_i
                    })
                    policy_matches[policy_j].append({
                        'opponent': policy_i,
                        'score': s_j
                    })

        # Now, for each policy, update their ratings based on all matches
        for policy in self.policy_names:
            matches = policy_matches[policy]
            if not matches:
                # No matches, proceed to next
                continue

            # Get current rating parameters
            mu = ratings[policy]['mu']
            phi = ratings[policy]['phi']
            sigma = ratings[policy]['sigma']

            # Compute v (variance), delta, and update factors
            v_inv = 0.0
            delta = 0.0

            for match in matches:
                opponent = match['opponent']
                s = match['score']
                mu_j = ratings[opponent]['mu']
                phi_j = ratings[opponent]['phi']
                g_phi_j = 1 / np.sqrt(1 + 3 * phi_j**2 / (np.pi**2))
                E_s = 1 / (1 + np.exp(-g_phi_j * (mu - mu_j)))
                v_inv += (g_phi_j**2) * E_s * (1 - E_s)
                delta += g_phi_j * (s - E_s)

            v = 1 / v_inv
            delta = v * delta

            # Update volatility σ'
            a = np.log(sigma**2)
            tau = self.tau
            epsilon = self.epsilon

            # Define function f(x)
            def f(x):
                exp_x = np.exp(x)
                numerator = exp_x * (delta**2 - phi**2 - v - exp_x)
                denominator = 2 * (phi**2 + v + exp_x)**2
                return (numerator / denominator) - ((x - a) / (tau**2))

            # Initial bounds
            A = a
            if delta**2 > phi**2 + v:
                B = np.log(delta**2 - phi**2 - v)
            else:
                k = 1
                while f(a - k * tau) < 0:
                    k += 1
                B = a - k * tau

            # Bisection method (Illinois algorithm) to solve for σ'
            fA = f(A)
            fB = f(B)
            iterations = 0
            while abs(B - A) > epsilon and iterations < self.max_iterations:
                C = A + (A - B) * fA / (fB - fA)
                fC = f(C)
                if fC * fB < 0:
                    A = B
                    fA = fB
                else:
                    fA /= 2
                B = C
                fB = fC
                iterations += 1

            if iterations >= self.max_iterations:
                # Handle non-convergence gracefully
                sigma_prime = np.exp(A / 2)
            else:
                sigma_prime = np.exp(A / 2)

            # Update φ'
            phi_star = np.sqrt(phi**2 + sigma_prime**2)
            phi_prime = 1 / np.sqrt(1 / (phi_star**2) + 1 / v)

            # Update μ'
            mu_prime = mu + phi_prime**2 * delta

            # Update ratings
            ratings[policy]['mu'] = mu_prime
            ratings[policy]['phi'] = phi_prime
            ratings[policy]['sigma'] = sigma_prime
            ratings[policy]['matches'] += len(matches)

        # After processing all matches, convert μ and φ back to ratings and RD
        # rating = μ * 173.7178 + 1500
        # RD = φ * 173.7178

        # Store the final ratings
        results = {}
        for policy in self.policy_names:
            mu = ratings[policy]['mu']
            phi = ratings[policy]['phi']
            rating = mu * 173.7178 + 1500
            RD = phi * 173.7178
            results[policy] = {
                'rating': rating,
                'RD': RD,
                'sigma': ratings[policy]['sigma'],
                'matches': ratings[policy]['matches']
            }

        #calculate summary verbose metrics
        for policy in self.policy_names:
            matches = policy_matches[policy]
            if not matches:
                # No matches, proceed to next
                continue
            verbose_results[policy]['avg_score'] = verbose_results[policy]['total_score'] / len(matches)
            verbose_results[policy]['win_pct'] = verbose_results[policy]['wins'] / len(matches)
        #calculate rank by win pct and add glicko scores to aux_metrics
        sorted_aux_metrics = sorted(verbose_results.items(), key=lambda x: x[1]['win_pct'], reverse=True)
        for i, (policy, metrics) in enumerate(sorted_aux_metrics):
            metrics['rank'] = i + 1
            metrics['glicko2'] = results[policy]['rating']

        if verbose:
            return results, verbose_results
        else:
            return results

    def format_results(self, results: Dict[str, Any]):
        # Headers
        headers = ['Policy', 'Glicko-2 Rating', 'Rating Deviation', 'Matches Played']
        data_rows = []

        ratings = [data['rating'] for data in results.values()]
        max_rating = max(ratings)
        min_rating = min(ratings)

        for policy in self.policy_names:
            rating = results[policy]['rating']
            RD = results[policy]['RD']
            matches = results[policy]['matches']
            rating_rounded = round(rating)
            RD_rounded = round(RD)
            if rating == max_rating:
                rating_str = colored(f"{rating_rounded}", "green")
            elif rating == min_rating:
                rating_str = colored(f"{rating_rounded}", "red")
            else:
                rating_str = colored(f"{rating_rounded}", "yellow")
            data_rows.append([policy, rating_str, RD_rounded, matches])

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
    historical_data = {}
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r") as file:
            print(f"Loading historical data from {scores_path}")
            try:
                historical_data = json.load(file)
            except json.JSONDecodeError:
                print(f"Failed to load historical data from {scores_path}")
            test.withHistoricalData(historical_data)

    results = test.evaluate()
    formatted_results = test.format_results(results)

    if scores_path:
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as file:
            print(f"Saving updated historical data to {scores_path}")
            historical_data.update(results)
            json.dump(historical_data, file, indent=4)

    return results, formatted_results
