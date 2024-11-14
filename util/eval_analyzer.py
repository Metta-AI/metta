import sys
from util.stats_library import *

eval_methods = {
    '1v1': MannWhitneyUTest,
    'elo_1v1': EloTest,
    'glicko2_1v1': Glicko2Test,
    'multiplayer': KruskalWallisTest,
}

stat_category_lookup = {
    # Existing category definitions remain unchanged
    'action.use.altar': ['action.use.altar'],
    'action.use': ['action.use'],
    'altar': ['action.use.altar'],
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
    def __init__(self, data: list, eval_method: str, stat_category: str, **kwargs):
        self.data = data
        self.eval_method = eval_method
        self.stat_category = stat_category
        self.categories_list = stat_category_lookup[stat_category]
        self.policy_names = []
        self.stats = {}
        self._prepare_data()
        test_class = eval_methods[self.eval_method]
        self.test_instance = test_class(self.stats, self.policy_names, self.categories_list, **kwargs)
        self.test_instance.run_test()

            # For policies not in this episode, their stat remains None

    def get_results(self):
        return self.test_instance.get_results()

    def get_display_results(self):
        return self.test_instance.get_formatted_results()

    def get_verbose_results(self):
        if hasattr(self.test_instance, 'get_verbose_results'):
            return self.test_instance.get_verbose_results()
        else:
            raise NotImplementedError(f"The eval_method {self.eval_method} does not currently support verbose results.")

    def get_updated_historicals(self):
        if hasattr(self.test_instance, 'get_updated_historicals'):
            return self.test_instance.get_updated_historicals()
        else:
            raise NotImplementedError(f"The eval_method {self.eval_method} does not currently support updating historical scores.")
