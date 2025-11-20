from typing import override

from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.procedural import MachinaArenaVariant
from cogames.cogs_vs_clips.sites import MACHINA_1


class ScatteredResourcesVariant(MachinaArenaVariant):
    name: str = "scattered_resources"
    description: str = "Scatter assemblers, chests, and chargers around the map."

    # Weights for the scattered items
    assembler_weight: float = 0.05
    chest_weight: float = 0.05
    charger_weight: float = 0.1

    # Additional building coverage to add for these items
    extra_coverage: float = 0.02

    @override
    def modify_node(self, node):
        extras = ["assembler", "chest", "charger"]
        current_names = node.building_names or [
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
            "charger",
        ]
        node.building_names = list(set(current_names + extras))

        if node.building_weights is None:
            node.building_weights = {}

        defaults = {
            "assembler": self.assembler_weight,
            "chest": self.chest_weight,
            "charger": self.charger_weight,
            "germanium_extractor": 0.1,
            "silicon_extractor": 0.1,
            "oxygen_extractor": 0.1,
            "carbon_extractor": 0.1,
        }

        # Ensure defaults are set if missing
        for k, v in defaults.items():
            if k not in node.building_weights:
                node.building_weights[k] = v

        # Boost the specific weights for our scattered items to ensure they appear
        node.building_weights["assembler"] = self.assembler_weight
        node.building_weights["chest"] = self.chest_weight
        node.building_weights["charger"] = self.charger_weight

        # Increase coverage slightly to account for more types
        if hasattr(node, "building_coverage"):
            node.building_coverage = max(node.building_coverage, self.extra_coverage)


MACHINA_TRAINING_MISSIONS: list[Mission] = [
    Mission(
        name="machina_scattered_easy",
        description="Machina 1 with frequent scattered assemblers and chests.",
        site=MACHINA_1,
        variants=[
            ScatteredResourcesVariant(
                assembler_weight=0.1, chest_weight=0.1, charger_weight=0.2, extra_coverage=0.04
            )
        ],
    ),
    Mission(
        name="machina_scattered_medium",
        description="Machina 1 with sparse scattered assemblers and chests.",
        site=MACHINA_1,
        variants=[
            ScatteredResourcesVariant(
                assembler_weight=0.02, chest_weight=0.02, charger_weight=0.1, extra_coverage=0.02
            )
        ],
    ),
]

