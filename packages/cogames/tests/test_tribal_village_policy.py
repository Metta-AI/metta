from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cogames.policy.tribal_village_policy import TribalPolicyEnvInfo
from mettagrid.policy.loader import discover_and_register_policies, resolve_policy_class_path


def test_policy_env_info_names_and_actions():
    info = TribalPolicyEnvInfo(
        observation_space=spaces.Box(0, 255, (3, 3, 3), dtype=np.uint8),
        action_space=spaces.Discrete(4),
        num_agents=2,
    )

    assert info.action_names == ["action_0", "action_1", "action_2", "action_3"]
    actions = info.actions
    assert len(actions) == 4
    assert actions[0].name == "action_0"


def test_policy_short_name_registration():
    discover_and_register_policies("cogames.policy")
    path = resolve_policy_class_path("tribal")
    assert path.endswith("TribalVillagePufferPolicy")
