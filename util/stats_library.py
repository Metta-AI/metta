from scipy.stats import mannwhitneyu
import numpy as np
from tabulate import tabulate
from termcolor import colored
    
def significance_and_effect(interpretations):
    all_positive = all(interpretation == "pos. effect" for interpretation in interpretations)
    all_negative = all(interpretation == "neg. effect" for interpretation in interpretations)
    any_positive = any(interpretation == "pos. effect" for interpretation in interpretations)
    any_negative = any(interpretation == "neg. effect" for interpretation in interpretations)
    all_no_effect = all(interpretation == "no effect" for interpretation in interpretations)

    # Case mapping based on effect direction and significance
    if all_positive:
        return colored("Greater than all!", "green")  # All positive effects
    elif all_negative:
        return colored("Less than all!", "red", attrs=["bold"]) # All negative effects
    elif any_positive and any_negative:
        return colored("Mixed results.", "yellow")  # Some positive, some negative
    elif any_positive and not any_negative:
        return colored("Some pos., none neg.", "yellow")  # Some positive, no negative
    elif any_negative and not any_positive:
        return colored("Some neg., none pos.", "yellow") # Some negative, no positive
    elif all_no_effect:
        return colored("No significance.", "yellow") # All no effect
    else:
        return "No interpretation."


def mann_whitney_u_test(stats, policy_names, categories_list):  
    """
    Perform the Mann-Whitney U test on the given episode scores for each stat in stat_list.

    Parameters:
    stats (dict): A dictionary where keys are stat names, and values are dictionaries
                  mapping policy names to lists of scores per episode.
    policy_names (list): List of policy names.
    stat_list (list): List of statistic names to analyze.

    Returns:
    list: A list of result dictionaries for each stat.

    Assumes that policy 1 plays in every episode.
    """
    results = []
    data_rows = []
    for stat_name in categories_list:
        if stat_name == 'policy_name':
            continue
        
        episode_scores = stats[stat_name]

        # Collect non-None scores per policy
        scores_per_policy = {}
        for policy in policy_names:
            scores = [score for score in episode_scores[policy] if score is not None]
            scores_per_policy[policy] = scores

        # Collect policy stats (mean, std_dev)
        policy_stats = []
        top_data_row = [stat_name]
        for policy in policy_names:
            scores = scores_per_policy[policy]
            if scores:
                if all(isinstance(score, (int, float)) for score in scores):
                    mean = np.mean(scores)
                    std_dev = np.std(scores, ddof=1)
                else:
                    mean = None
                    std_dev = None
                policy_stats.append({
                    'policy_name': policy,
                    'mean': mean,
                    'std_dev': std_dev
                })
                top_data_row.append(f"{mean:.1f} ± {std_dev:.1f}" if mean is not None else "None ± None")
            else:
                policy_stats.append({
                    'policy_name': policy,
                    'mean': None,
                    'std_dev': None
                })
                top_data_row.append("None ± None")

        policy1 = policy_names[0]
        comparison_scores = [[[], []] for _ in range(len(policy_names) - 1)]
        for idx, policy in enumerate(policy_names[1:], start=1):
            # Loop through the scores from policy1 and the current policy together
            for score1, score2 in zip(episode_scores[policy1], episode_scores[policy]):
                if score1 is not None and score2 is not None:
                    comparison_scores[idx-1][0].append(score1)  # Append policy1's score
                    comparison_scores[idx-1][1].append(score2)  # Append the current policy's score

        lower_data_row = ['', '']
        comparisons = []
        interpretations = []
        for idx, comparison in enumerate(comparison_scores):
            policy1_scores = comparison[0]
            policy2_scores = comparison[1]
            interpretation = None
            if len(policy1_scores) == len(policy2_scores) and len(policy1_scores) > 0:
                #for some reason, mannwhitneyu seems to want Y followed by X. Otherwise it returns U2 
                u_statistic, p_value = mannwhitneyu(policy2_scores, policy1_scores)
                n1 = len(policy1_scores)
                n2 = len(policy2_scores)
                # Compute rank-biserial correlation coefficient as effect size
                r = 1 - (2 * u_statistic) / (n1 * n2)
                comparisons.append({
                    'policy1': policy1,
                    'policy2': policy_names[idx],
                    'effect_size': r,
                    'p_value': p_value,
                    'interpretation': interpretation
                })
                if r > 0 and p_value < 0.05:
                    interpretation = "pos. effect"
                    r = colored(f"{r:.2f}", 'green')
                    p_value = colored(f"{p_value:.2f}", 'green')
                elif r < 0 and p_value < 0.05:
                    interpretation = "neg. effect"
                    r = colored(f"{r:.2f}", 'red')
                    p_value = colored(f"{p_value:.2f}", 'red')
                else:
                    interpretation = "no effect"
                    r = str(round(r, 2))
                    p_value = str(round(p_value, 2))
            else:
                comparisons.append({
                    'policy1': policy1,
                    'policy2': policy_names[idx],
                    'effect_size': None,
                    'p_value': None,
                    'interpretation ': interpretation
                })
            interpretations.append(interpretation)
            lower_data_row.append(f"{p_value}, {r}")

        summary_value = significance_and_effect(interpretations)
        lower_data_row[1] = summary_value

        results.append({
            'stat_name': stat_name,
            'policy_stats': policy_stats,
            'comparisons': comparisons,
            'summary_value': summary_value
        })

        data_rows.append(top_data_row)
        data_rows.append(lower_data_row)

    headers = ['']
    for policy_name in policy_names:
        header = f"{policy_name}\n(mean ± std)\n(p-val, effect size)"
        headers.append(header)
    
    print(tabulate(data_rows, headers=headers, tablefmt="fancy_grid"))

    return results

def elo_test(stats, policy_names, categories_list):
    """
    Perform the 1v1 Elo test on the given episode scores for each stat in stat_list.
    Assumes all games are 1v1.
    Assumes policy 1 plays in every episode.
    """
    winning_margin = 10 # set a minimum winning margin for a win. If less than this, it's a tie.
    all_scores = []
    for policy in policy_names:
        all_scores.append(stats['action.use.energy.altar'][policy])
    total_episodes = len(all_scores[0])

    all_scores = stats['action.use.energy.altar']

    elo = [1000] * len(policy_names)
    #note, apparently is standard to start w 1,000 but I find that starting at 0 shows unanimous otcomes more clearly

    def get_elo_constant(e):
        #dynamic elo constant based on episode number
        e = e+1 #episode number, starting from 1
        Ki = 32 # initial constant
        Kf = 2 # final constant
        return Ki + (Kf - Ki) / (total_episodes - 1) * (e - 1)

    for episode_no in range(total_episodes):
        K = get_elo_constant(episode_no)
        policy1_score = all_scores[policy_names[0]][episode_no]
        for p2_idx_minus_1, policy_name in enumerate(policy_names[1:]):
            policy2_score = all_scores[policy_name][episode_no]
            if policy1_score is not None and policy2_score is not None:
                E_A = 1 / (1 + 10 ** ((elo[p2_idx_minus_1+1] - elo[0]) / 400))
                E_B = 1- E_A
                if (policy1_score - policy2_score) > winning_margin:
                    S_A = 1
                    S_B = 0
                elif (policy2_score - policy1_score) > winning_margin:
                    S_A = 0
                    S_B = 1
                else:
                    S_A = 0.5
                    S_B = 0.5
                elo[0] = elo[0] + K * (S_A - E_A)
                elo[p2_idx_minus_1+1] = elo[p2_idx_minus_1+1] + K * (S_B - E_B)

    headers = ['']
    for policy_name in policy_names:
        header = f"{policy_name}\nElo"
        headers.append(header)
    
    # formatted_elo = ['Elo:']
    # for elo_val in elo:
    #     formatted_elo.append(f"{elo_val:.2f}")

    # Start formatting with 'Elo:'
    formatted_elo = ['Elo:']

    # Handle the color of policy1's score 
    max_elo = max(elo)
    min_elo = min(elo)
    # first_elo_rounded = round(elo[0])
    if elo[0] == max_elo:
        formatted_elo.append(colored(f"{round(elo[0])}", "green"))
    elif elo[0] == min_elo:
        formatted_elo.append(colored(f"{round(elo[0])}", "red"))
    else:
        formatted_elo.append(colored(f"{round(elo[0])}", "yellow"))

    # Handle the rest of the elements (rounded)
    for elo_val in elo[1:]:
        formatted_elo.append(f"{round(elo_val)}")
    
    data_rows = [formatted_elo]

    print(tabulate(data_rows, headers=headers, tablefmt="fancy_grid"))


def kruskal_wallis_test(stats, policy_names, categories_list):
    """
    Function to perform Kruskal-Wallis test. Currently not implemented.
    """
    pass

def glicko2_test(stats, policy_names, categories_list):
    """
    Function to perform Glicko-2 test. Currently not implemented.
    """
    pass