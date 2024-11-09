from scipy.stats import mannwhitneyu
import numpy as np
from tabulate import tabulate
from termcolor import colored

class StatisticalTest:
    def __init__(self, stats, policy_names, categories_list, **kwargs):
        self.stats = stats
        self.policy_names = policy_names
        self.categories_list = categories_list
        self.results = None  # Dictionary results for manipulation or for saving
        self.formatted_results = None # For printing to console
        self.params = kwargs

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

# helper function to update historical scores for Elo and Glicko2 tests
def update_scores(historical_scores, new_scores):
    for policy, scores in new_scores.items():
        if policy in historical_scores:
            historical_scores[policy].update(scores)
        else:
            historical_scores[policy] = scores
    return historical_scores

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
    def __init__(self, stats, policy_names, categories_list, **kwargs):
        super().__init__(stats, policy_names, categories_list, **kwargs)
        self.initial_elo_scores = self.params.get('initial_elo_scores', None)

        # Initialize elo scores and episodes
        if self.initial_elo_scores is None:
            # Initialize with default values
            self.elo_scores = {policy: {'score': 1000, 'matches': 0} for policy in self.policy_names}
        else:
            # Initialize with provided values, defaulting missing entries
            self.elo_scores = {}
            for policy in self.policy_names:
                self.elo_scores[policy] = self.initial_elo_scores.get(policy, {'score': 1000, 'matches': 0})

    def run_test(self):
        winning_margin = 1  # Set a minimum winning margin for a win
        # Use the first stat in categories_list instead of hardcoding 'altar'
        stat_name = self.categories_list[0]
        all_scores = self.stats[stat_name]
        total_episodes = len(next(iter(all_scores.values())))

        def get_k(policy):
            n = self.elo_scores[policy]['matches']
            # Adjust K based on the number of episodes played by this policy
            if n < 30:
                return 32
            elif n < 100:
                return 16
            else:
                return 8

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
                        elo_i = self.elo_scores[policy_i]['score']
                        elo_j = self.elo_scores[policy_j]['score']

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
                        K_i = get_k(policy_i)
                        K_j = get_k(policy_j)

                        # Update elo ratings
                        self.elo_scores[policy_i]['score'] += K_i * (S_i - E_i)
                        self.elo_scores[policy_j]['score'] += K_j * (S_j - E_j)

                        # Update episodes played per policy
                        self.elo_scores[policy_i]['matches'] += 1
                        self.elo_scores[policy_j]['matches'] += 1

        # Output updated elo scores and episodes played
        self.results = self.elo_scores

    def get_updated_historicals(self):
        if self.initial_elo_scores is not None:
            self.updated_scores = update_scores(self.initial_elo_scores, self.results)
            return self.updated_scores
        return None 

    def format_results(self):
        # Headers
        headers = ['Policy', 'Elo Rating', 'Matches Played']
        data_rows = []

        elo_values = [data['score'] for data in self.results.values()]
        max_elo = max(elo_values)
        min_elo = min(elo_values)

        for policy in self.policy_names:
            elo_rating = self.results[policy]['score']
            episodes_played = self.results[policy]['matches']
            elo_rounded = round(elo_rating)
            if elo_rating == max_elo:
                elo_str = colored(f"{elo_rounded}", "green")
            elif elo_rating == min_elo:
                elo_str = colored(f"{elo_rounded}", "red")
            else:
                elo_str = colored(f"{elo_rounded}", "yellow")
            data_rows.append([policy, elo_str, episodes_played])

        self.formatted_results = tabulate(data_rows, headers=headers, tablefmt="fancy_grid")

# Placeholder classes for KruskalWallisTest for multiplayer significance analysis
class KruskalWallisTest(StatisticalTest):
    def run_test(self):
        # Not implemented yet
        pass

    def format_results(self):
        # Not implemented yet
        pass

class Glicko2Test(StatisticalTest):
    def __init__(self, stats, policy_names, categories_list, **kwargs):
        super().__init__(stats, policy_names, categories_list, **kwargs)
        # Initialize ratings, deviations, and volatilities per policy
        initial_mu = 0.0 # by definition
        initial_phi = 2.014761872416068 # φ = RD / 173.7178, RD = 350 ('tis what Mark Glickman used)
        initial_sigma = 0.06 # Starting volatility for new players. Can adjust.
        self.tau = self.params.get('tau', 1.1) # Sets how much volatility sigma gets updated. I think we need a lot of games or more dominance for this to be meaningful.
        self.epsilon = self.params.get('epsilon', 0.000001) # Exposed in case it hangs on the Illinois algo, you can feed it a greater epsilon to try again.
        self.max_iterations = self.params.get('max_iterations', 10) # Same as above, in case it hangs on the Illinois algo.
        self.ratings = {}
        self.initial_ratings = self.params.get('initial_glicko2_scores', {})
        print(f"Tau: {self.tau}")

        for policy in policy_names:
            # Use initial ratings if provided, else default values
            if policy in self.initial_ratings:
                rating_info = self.initial_ratings[policy]
                mu = (rating_info['rating'] - 1500) / 173.7178  # Convert rating to mu
                phi = rating_info['RD'] / 173.7178  # Convert RD to phi
                sigma = rating_info.get('sigma', initial_sigma)
                matches = rating_info.get('matches', 0)
            else:
                mu = initial_mu
                phi = initial_phi
                sigma = initial_sigma
                matches = 0

            self.ratings[policy] = {
                'mu': mu,
                'phi': phi,
                'sigma': sigma,
                'matches': matches
            }

    def run_test(self):
        # Collect all match results over all episodes
        stat_name = self.categories_list[0]
        all_scores = self.stats[stat_name]
        total_episodes = len(next(iter(all_scores.values())))

        # For each policy, collect matches over all episodes
        policy_matches = {policy: [] for policy in self.policy_names}
        # Extra metrics for checking if Glicko is doing the job
        self.verbose_results = {policy: {'total_score': 0, 
                                'wins': 0, 
                                'losses': 0, 
                                'win_pct': 0,
                                'avg_score': 0, 
                                'rank': 0,
                                'glicko2': 0} for policy in self.policy_names}

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
                    self.verbose_results[policy_i]['total_score'] += score_i
                    self.verbose_results[policy_j]['total_score'] += score_j
                    if s_i == 1:
                        self.verbose_results[policy_i]['wins'] += 1
                        self.verbose_results[policy_j]['losses'] += 1
                    elif s_j == 1:
                        self.verbose_results[policy_j]['wins'] += 1
                        self.verbose_results[policy_i]['losses'] += 1
                        
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
            mu = self.ratings[policy]['mu']
            phi = self.ratings[policy]['phi']
            sigma = self.ratings[policy]['sigma']

            # Compute v (variance), delta, and update factors
            v_inv = 0.0
            delta = 0.0

            for match in matches:
                opponent = match['opponent']
                s = match['score']
                mu_j = self.ratings[opponent]['mu']
                phi_j = self.ratings[opponent]['phi']
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
            self.ratings[policy]['mu'] = mu_prime
            self.ratings[policy]['phi'] = phi_prime
            self.ratings[policy]['sigma'] = sigma_prime
            self.ratings[policy]['matches'] += len(matches)

        # After processing all matches, convert μ and φ back to ratings and RD
        # rating = μ * 173.7178 + 1500
        # RD = φ * 173.7178

        # Store the final ratings
        self.results = {}
        for policy in self.policy_names:
            mu = self.ratings[policy]['mu']
            phi = self.ratings[policy]['phi']
            rating = mu * 173.7178 + 1500
            RD = phi * 173.7178
            self.results[policy] = {
                'rating': rating,
                'RD': RD,
                'sigma': self.ratings[policy]['sigma'],
                'matches': self.ratings[policy]['matches']
            }

        
        #calculate summary verbose metrics
        for policy in self.policy_names:
            matches = policy_matches[policy]
            if not matches:
                # No matches, proceed to next
                continue
            self.verbose_results[policy]['avg_score'] = self.verbose_results[policy]['total_score'] / len(matches)
            self.verbose_results[policy]['win_pct'] = self.verbose_results[policy]['wins'] / len(matches)
        #calculate rank by win pct and add glicko scores to aux_metrics
        sorted_aux_metrics = sorted(self.verbose_results.items(), key=lambda x: x[1]['win_pct'], reverse=True)
        for i, (policy, metrics) in enumerate(sorted_aux_metrics):
            metrics['rank'] = i + 1
            metrics['glicko2'] = self.results[policy]['rating']

    def get_verbose_results(self):
        return self.verbose_results
        
    def get_updated_historicals(self):
        if self.initial_ratings is not None:
            self.updated_scores = update_scores(self.initial_ratings, self.results)
            return self.updated_scores
        return None

    def format_results(self):
        # Headers
        headers = ['Policy', 'Glicko-2 Rating', 'Rating Deviation', 'Matches Played']
        data_rows = []

        ratings = [data['rating'] for data in self.results.values()]
        max_rating = max(ratings)
        min_rating = min(ratings)

        for policy in self.policy_names:
            rating = self.results[policy]['rating']
            RD = self.results[policy]['RD']
            matches = self.results[policy]['matches']
            rating_rounded = round(rating)
            RD_rounded = round(RD)
            if rating == max_rating:
                rating_str = colored(f"{rating_rounded}", "green")
            elif rating == min_rating:
                rating_str = colored(f"{rating_rounded}", "red")
            else:
                rating_str = colored(f"{rating_rounded}", "yellow")
            data_rows.append([policy, rating_str, RD_rounded, matches])

        self.formatted_results = tabulate(data_rows, headers=headers, tablefmt="fancy_grid")
