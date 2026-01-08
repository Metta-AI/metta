import random

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.simulator.interface import AgentObservation

from .types import BaselineHyperparameters, CellType, SimpleAgentState
from .utils import (
    change_vibe_action,
    is_station,
    is_wall,
    parse_observation,
    read_inventory_from_obs,
    update_agent_position,
    use_object_at,
)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class DemoPolicyImpl(StatefulPolicyImpl[SimpleAgentState]):
    def __init__(self, policy_env_info, agent_id, hyperparams, *, heart_recipe=None):
        self._agent_id = agent_id
        self._hyperparams = hyperparams
        self._policy_env_info = policy_env_info
        self._actions = policy_env_info.actions
        self._move_deltas = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}

        self._obs_hr = policy_env_info.obs_height // 2
        self._obs_wr = policy_env_info.obs_width // 2

        if heart_recipe:
            self._heart_recipe = {k: v for k, v in heart_recipe.items() if k != "energy"}
        else:
            self._heart_recipe = None

    def initial_agent_state(self):
        center = 25
        return SimpleAgentState(
            agent_id=self._agent_id,
            map_height=50,
            map_width=50,
            occupancy=[[CellType.FREE.value] * 50 for _ in range(50)],
            row=center,
            col=center,
            heart_recipe=self._heart_recipe,
        )

    # ------------------------------------------------------------
    # Utility helpers (kept tiny)
    # ------------------------------------------------------------

    def _adjacent(self, s, pos):
        return manhattan((s.row, s.col), pos) == 1

    def _random_step(self, s, parsed):
        dirs = list(self._move_deltas.keys())
        random.shuffle(dirs)
        blocked = {
            (r, c)
            for (r, c), obj in parsed.nearby_objects.items()
            if self._adjacent(s, (r, c))
            and (
                is_wall(obj.name)
                or "extractor" in obj.name
                or is_station(obj.name, "assembler")
                or is_station(obj.name, "chest")
                or is_station(obj.name, "charger")
                or (obj.name == "agent" and obj.agent_group != s.agent_id)
            )
        }
        for d in dirs:
            dr, dc = self._move_deltas[d]
            nr, nc = s.row + dr, s.col + dc
            if (nr, nc) not in blocked:
                return self._actions.move.Move(d)
        return self._actions.noop.Noop()

    def _step_towards(self, s, target, parsed):
        """Single-step greedy pursuit, else random."""
        r, c = s.row, s.col
        tr, tc = target
        cand = []
        if abs(tr - r) >= abs(tc - c):
            if tr < r:
                cand.append("north")
            elif tr > r:
                cand.append("south")
            if tc < c:
                cand.append("west")
            elif tc > c:
                cand.append("east")
        else:
            if tc < c:
                cand.append("west")
            elif tc > c:
                cand.append("east")
            if tr < r:
                cand.append("north")
            elif tr > r:
                cand.append("south")

        blocked = {
            (rr, cc)
            for (rr, cc), obj in parsed.nearby_objects.items()
            if self._adjacent(s, (rr, cc))
            and (
                is_wall(obj.name)
                or "extractor" in obj.name
                or is_station(obj.name, "assembler")
                or is_station(obj.name, "chest")
                or is_station(obj.name, "charger")
                or (obj.name == "agent" and obj.agent_group != s.agent_id)
            )
        }

        for d in cand:
            dr, dc = self._move_deltas[d]
            nr, nc = r + dr, c + dc
            if (nr, nc) not in blocked:
                return self._actions.move.Move(d)

        return self._random_step(s, parsed)

    def _closest(self, s, parsed, pred):
        items = [pos for pos, obj in parsed.nearby_objects.items() if pred(obj)]
        return min(items, key=lambda p: manhattan((s.row, s.col), p)) if items else None

    def _rtype(self, name):
        name = name.lower().replace("clipped_", "")
        if "_extractor" not in name:
            return None
        name = name.replace("_extractor", "")
        return name if name in ("carbon", "oxygen", "germanium", "silicon") else None

    # ------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------
    def step_with_state(self, obs: AgentObservation, s: SimpleAgentState):
        s.step_count += 1
        read_inventory_from_obs(s, obs, obs_hr=self._obs_hr, obs_wr=self._obs_wr)
        update_agent_position(s, move_deltas=self._move_deltas)

        parsed = parse_observation(
            s,
            obs,
            obs_hr=self._obs_hr,
            obs_wr=self._obs_wr,
            spatial_feature_names={"tag", "cooldown_remaining", "clipped", "remaining_uses"},
            agent_feature_key_by_name={"agent:group": "agent_group", "agent:frozen": "agent_frozen"},
            protocol_input_prefix="protocol_input:",
            protocol_output_prefix="protocol_output:",
            tag_names=self._policy_env_info.tag_id_to_name,
        )

        # Learn recipe if visible
        if s.heart_recipe is None:
            for _pos, obj in parsed.nearby_objects.items():
                if obj.name == "assembler" and obj.protocol_outputs.get("heart", 0) > 0:
                    s.heart_recipe = {k: v for k, v in obj.protocol_inputs.items() if k != "energy"}

        # ---------------- PRE-PHASE: find assembler to learn recipe ----------------
        if s.heart_recipe is None:
            if s.current_glyph != "heart_a":
                s.current_glyph = "heart_a"
                return change_vibe_action("heart_a", actions=self._actions), s

            assembler = self._closest(s, parsed, lambda o: is_station(o.name.lower(), "assembler"))
            if assembler:
                if self._adjacent(s, assembler):
                    return use_object_at(s, assembler, actions=self._actions, move_deltas=self._move_deltas), s
                return self._step_towards(s, assembler, parsed), s

            return self._random_step(s, parsed), s

        # ---------------- MAIN PHASE ----------------

        # Deliver hearts
        if s.hearts > 0:
            chest = self._closest(s, parsed, lambda o: is_station(o.name.lower(), "chest"))
            if chest:
                if s.current_glyph != "default":
                    s.current_glyph = "default"
                    return change_vibe_action("default", actions=self._actions), s
                if self._adjacent(s, chest):
                    return use_object_at(s, chest, actions=self._actions, move_deltas=self._move_deltas), s
                return self._step_towards(s, chest, parsed), s

        # Assemble
        if (
            s.carbon >= s.heart_recipe.get("carbon", 0)
            and s.oxygen >= s.heart_recipe.get("oxygen", 0)
            and s.germanium >= s.heart_recipe.get("germanium", 0)
            and s.silicon >= s.heart_recipe.get("silicon", 0)
        ):
            assembler = self._closest(s, parsed, lambda o: is_station(o.name.lower(), "assembler"))
            if assembler:
                if s.current_glyph != "heart_a":
                    s.current_glyph = "heart_a"
                    return change_vibe_action("heart_a", actions=self._actions), s
                if self._adjacent(s, assembler):
                    return use_object_at(s, assembler, actions=self._actions, move_deltas=self._move_deltas), s
                return self._step_towards(s, assembler, parsed), s

        # Gather needed resources
        deficits = {
            r: s.heart_recipe.get(r, 0) - getattr(s, r, 0) for r in ("carbon", "oxygen", "germanium", "silicon")
        }
        needed = [
            (pos, obj, self._rtype(obj.name.lower()))
            for pos, obj in parsed.nearby_objects.items()
            if "extractor" in obj.name.lower()
        ]

        needed = [(pos, obj, r) for pos, obj, r in needed if r and deficits[r] > 0]

        if needed:
            pos, obj, r = min(needed, key=lambda x: manhattan((s.row, s.col), x[0]))
            if self._adjacent(s, pos):
                return use_object_at(s, pos, actions=self._actions, move_deltas=self._move_deltas), s
            return self._step_towards(s, pos, parsed), s

        # Otherwise wander
        return self._random_step(s, parsed), s


class DemoPolicy(MultiAgentPolicy):
    short_names = ["tiny_baseline"]

    def __init__(self, policy_env_info, device: str = "cpu", hyperparams=None, *, heart_recipe=None):
        super().__init__(policy_env_info, device=device)
        self._hyperparams = hyperparams or BaselineHyperparameters()
        self._heart_recipe = heart_recipe
        self._agent_policies = {}

    def agent_policy(self, agent_id):
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                DemoPolicyImpl(self._policy_env_info, agent_id, self._hyperparams, heart_recipe=self._heart_recipe),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
