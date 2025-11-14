"""Convert MettaGrid tokenized observations to text."""

import json

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation


class ObservationEncoder:
    """Convert MettaGrid tokenized observations to text."""

    def __init__(self, policy_env_info: PolicyEnvInterface):
        self.policy_env_info = policy_env_info
        self.feature_id_to_name = {feat.id: feat.name for feat in policy_env_info.obs_features}

    def encode(self, obs: AgentObservation) -> str:
        """Encode observation as structured JSON."""
        # Parse tokens into structured data
        agent_state = {}
        visible_entities = []
        inventory = {}

        for token in obs.tokens:
            # token.raw_token is [coord, feature_id, value]
            if len(token.raw_token) < 3:
                continue

            coord, feature_id, value = token.raw_token[:3]

            # Skip padding tokens
            if coord == 255:
                continue

            # Get feature name
            feature_name = self.feature_id_to_name.get(feature_id, f"unknown_{feature_id}")

            # Decode coordinate
            x = coord & 0x0F  # Low nibble
            y = (coord >> 4) & 0x0F  # High nibble

            # Categorize by feature type
            if "health" in feature_name.lower():
                agent_state["health"] = int(value)
            elif "position" in feature_name.lower() or (x == 0 and y == 0):
                agent_state["position"] = {"x": int(x), "y": int(y)}
            elif any(item in feature_name.lower() for item in ["ore", "battery", "laser", "armor"]):
                inventory[feature_name] = int(value)
            else:
                # Visible entities
                visible_entities.append(
                    {"type": feature_name, "position": {"x": int(x), "y": int(y)}, "value": int(value)}
                )

        # Build observation dict
        obs_dict = {}
        if agent_state:
            obs_dict["agent_state"] = agent_state
        if inventory:
            obs_dict["inventory"] = inventory
        if visible_entities:
            # Limit visible entities to avoid token explosion
            obs_dict["visible_entities"] = visible_entities[:20]

        return json.dumps(obs_dict, separators=(",", ":"))  # Compact JSON

    def encode_with_return(self, obs: AgentObservation, target_return: float) -> str:
        """Encode observation with target return for Decision Transformer."""
        obs_json = self.encode(obs)
        prompt = f"Target return: {target_return:.1f}\nObservation: {obs_json}"
        return prompt
