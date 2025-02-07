#%%
import os
os.chdir("/Users/daphne/Desktop/stemai/p/metta/")
from rl.wandb.wanduckdb import WandbDuckDB
from util.stats_library import MannWhitneyUTest, EloTest, Glicko2Test, get_test_results
from typing import List, Optional

def prepare_data_for_stats(wandb_db, metrics: List[str], policies: Optional[List[str]] = None) -> List[List[dict]]:
    """Convert WandB data into format expected by statistical tests with optional policy filtering."""
    # Get all episodes
    query = "SELECT DISTINCT episode_index FROM eval_stats ORDER BY episode_index"
    episodes = wandb_db.query(query)['episode_index'].tolist()

    # Initialize data structure
    data = []

    # Build policy filter if provided
    policy_filter = ""
    if policies:
        policy_list = ", ".join([f"'{policy}'" for policy in policies])
        policy_filter = f"AND policy_name IN ({policy_list})"

    # For each episode, get all agent data
    for episode_idx in episodes:
        episode_query = f"""
        SELECT policy_name, {', '.join(metrics)}
        FROM eval_stats
        WHERE episode_index = {episode_idx}
        {policy_filter}
        """
        episode_data = wandb_db.query(episode_query).to_dict('records')
        if episode_data:  # Only include episodes with matching data
            data.append(episode_data)

    return data

#%%
entity = "metta-research"
project = "metta"
artifact_name = "daphne_eval"

# Instantiate the class; this will download the artifact and load the JSON into DuckDB.
wandb_db = WandbDuckDB(entity, project, artifact_name, table_name="eval_stats")

#table will have num_episodes * num_agents rows, one per episode per agent

# %%
query = "SELECT * FROM eval_stats"
wandb_db.query(query)

# %%
#reward per episode per policy
print(f"Reward per episode per policy:")
reward_result = wandb_db.metric_per_episode_per_policy("agent.action.use.altar")
reward_result
#%%
#reward per policy
print(f"Reward per policy:")
reward_result = wandb_db.average_metrics_by_policy(["agent.action.use.altar"])
reward_result
#%%
# %%

first_use_fields = wandb_db.get_metrics_by_pattern("action.use.first_use")
print(first_use_fields)
first_use_result = wandb_db.average_metrics_by_policy(first_use_fields)
first_use_result
# %%

for test_type in ["glicko2", "elo", "mann_whitney"]:

    policies = ["b.daveey.train.sm.teams.100:v0", "b.daveey.train.sm.teams.75:v0"]
    mode = "sum"
    label = None
    scores_path = None
    categories = ['"agent.action.use.altar"']

    print(f"\nRunning {test_type} test for metrics: {categories}")

    # Prepare data for statistical tests
    data = prepare_data_for_stats(wandb_db, categories, policies)

    # Initialize appropriate test
    if test_type == "mann_whitney":
        test = MannWhitneyUTest(data, categories, mode, label)
    elif test_type == "elo":
        test = EloTest(data, categories, mode, label)
    elif test_type == "glicko2":
        test = Glicko2Test(data, categories)
    else:
        print(f"Unknown test type: {test_type}")
        #continue


    # Run test and get results
    results, formatted_results = get_test_results(test, scores_path)
    print(f"\n{test_type} test results:")
    print(formatted_results)
    # %%
