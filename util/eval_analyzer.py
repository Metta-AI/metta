import sys
from util.stats_library import *

eval_methods = {
    '1v1': MannWhitneyUTest,
    'elo_1v1': EloTest,
    'glicko2_1v1': Glicko2Test,
    'multiplayer': KruskalWallisTest,
}

stat_category_lookup = {
    'altar': ['action.use.energy.altar'],
    'all': [
        "action.rotate.energy",
        "action.attack",
        "action.attack.energy",
        "action.move.energy",
        "action.gift.energy",
        "r3.stolen",
        "action.rotate",
        "action.attack.altar",
        "action.use.altar",
        "r1.stolen",
        "action.attack.agent",
        "shield_damage",
        "damage.altar",
        "status.shield.ticks",
        "action.use.energy.altar",
        "action.attack.wall",
        "destroyed.wall",
        "action.shield.energy",
        "r2.gained",
        "r3.gained",
        "action.move",
        "action.use.energy",
        "r2.stolen",
        "status.frozen.ticks",
        "shield_upkeep",
        "attack.frozen",
        "r1.gained",
        "action.use",
        "damage.wall",
    ],
    'adversarial': [
        "action.attack",
        "action.attack.energy",
        "action.attack.altar",
        "action.attack.agent",
        "action.attack.wall",
        "damage.altar",
        "damage.wall",
        "shield_damage",
        "attack.frozen",
        "r1.stolen",
        "r2.stolen",
        "r3.stolen",
        "destroyed.wall"
    ],
    'shield': [
        "shield_damage",
        "status.shield.ticks",
        "action.shield.energy",
        "shield_upkeep"
    ],
}

class Analysis:
    def __init__(self, data: list, eval_method: str, stat_category: str):
        self.data = data
        self.eval_method = eval_method
        self.stat_category = stat_category
        self.categories_list = stat_category_lookup[stat_category]
        self.policy_names = []
        self.stats = {}
        self.results = None
        self.test_instance = None  # Store the test instance
        self.prepare_data()

    def prepare_data(self):
        # Extract policy names
        for episode in self.data:
            for agent in episode:
                policy_name = agent.get('policy_name', "unknown")
                if policy_name and policy_name not in self.policy_names:
                    self.policy_names.append(policy_name)

        # Initialize stats dictionaries for each stat and policy with None values
        for stat_name in self.categories_list:
            self.stats[stat_name] = { policy_name: [None] * len(self.data) for policy_name in self.policy_names }

        # Extract stats per policy per episode
        for idx, episode in enumerate(self.data):
            # Keep track of which policies participated in this episode
            policies_in_episode = set()
            for agent in episode:
                policy = agent.get('policy_name', "unknown")
                if policy is None:
                    continue
                policies_in_episode.add(policy)
                # Loop through each stat and set this policy's stat for the episode
                for stat_name in self.categories_list:
                    stat_value = agent.get(stat_name, 0)
                    if self.stats[stat_name][policy][idx] is None:
                        self.stats[stat_name][policy][idx] = stat_value
                    else:
                        self.stats[stat_name][policy][idx] += stat_value
            # For policies not in this episode, their stat remains None

    def run_analysis(self):
        test_class = eval_methods[self.eval_method]
        self.test_instance = test_class(self.stats, self.policy_names, self.categories_list)
        self.test_instance.run_test()
        self.results = self.test_instance.get_results()

    def get_results(self):
        return self.results

    def display_results(self):
        if self.test_instance:
            formatted_output = self.test_instance.get_formatted_results()
            print(formatted_output)
        else:
            print("No analysis has been run yet.")
