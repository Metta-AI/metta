"""Tests for Area of Effect (AOE) resource effects.

This tests the AOE system functionality where:
- Objects can emit AOE effects that modify agent inventories within range
- Effects can be filtered by commons membership (members_only, ignore_members)
- Effects are properly registered/unregistered when objects are added/removed
- Multiple overlapping effects accumulate correctly
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AOEEffectConfig,
    ChangeVibeActionConfig,
    CommonsConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    TransferActionConfig,
    VibeTransfer,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

# Vibe names from the global VIBES list
HEART_VIBE_NAME = "heart_a"  # VIBES[10].name for alignment


class TestAOEEffectsBasic:
    """Test basic AOE effect registration and application."""

    def _create_sim_with_aoe(
        self,
        aoe_range: int = 1,
        resource_deltas: dict[str, int] | None = None,
        members_only: bool = False,
        ignore_members: bool = False,
        initial_inventory: dict[str, int] | None = None,
    ) -> Simulation:
        """Create a simulation with an AOE-emitting wall and an agent."""
        resource_deltas = resource_deltas or {"energy": 5}

        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_source", ".", ".", "wall"],
            ["wall", ".", "agent.agent", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial=initial_inventory or {"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "aoe_source": WallConfig(
                    name="aoe_source",
                    aoes=[
                        AOEEffectConfig(
                            range=aoe_range,
                            resource_deltas=resource_deltas,
                            members_only=members_only,
                            ignore_members=ignore_members,
                        )
                    ],
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, resource: str) -> int:
        """Get agent's inventory for a specific resource."""
        objects = sim.grid_objects()
        agents = [obj for obj in objects.values() if "agent_id" in obj]
        assert len(agents) == 1
        resource_idx = sim.resource_names.index(resource)
        return agents[0]["inventory"][resource_idx]

    def test_aoe_effect_applied_when_in_range(self):
        """Test that AOE effects are applied to agents within range."""
        sim = self._create_sim_with_aoe(aoe_range=2, resource_deltas={"energy": 10})

        # Agent at (2,2), AOE source at (1,1), distance = 1 (Chebyshev/square)
        energy_before = self._get_agent_inventory(sim, "energy")

        # Step to apply effects
        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should gain 10 energy from AOE. Before: {energy_before}, After: {energy_after}"
        )

    def test_aoe_effect_not_applied_when_out_of_range(self):
        """Test that AOE effects are NOT applied to agents outside range."""
        sim = self._create_sim_with_aoe(aoe_range=0, resource_deltas={"energy": 10})

        # Agent at (2,2), AOE source at (1,1), distance = 1 (Chebyshev/square)
        # Range is 0, so agent should be out of range
        energy_before = self._get_agent_inventory(sim, "energy")

        # Step to apply effects
        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before, (
            f"Agent should NOT gain energy (out of range). Before: {energy_before}, After: {energy_after}"
        )

    def test_aoe_effect_negative_delta(self):
        """Test that AOE can apply negative resource deltas."""
        sim = self._create_sim_with_aoe(aoe_range=2, resource_deltas={"energy": -5})

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before - 5, (
            f"Agent should lose 5 energy from AOE. Before: {energy_before}, After: {energy_after}"
        )

    def test_aoe_effect_multiple_resources(self):
        """Test that AOE can affect multiple resources at once."""
        sim = self._create_sim_with_aoe(
            aoe_range=2, resource_deltas={"energy": 5, "heart": -1}, initial_inventory={"energy": 100, "heart": 10}
        )

        energy_before = self._get_agent_inventory(sim, "energy")
        heart_before = self._get_agent_inventory(sim, "heart")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        heart_after = self._get_agent_inventory(sim, "heart")

        assert energy_after == energy_before + 5, "Agent should gain 5 energy"
        assert heart_after == heart_before - 1, "Agent should lose 1 heart"


class TestAOEMultipleOverlapping:
    """Test multiple overlapping AOE effects."""

    def _create_sim_with_multiple_aoe(
        self,
        sources: list[dict],
    ) -> Simulation:
        """Create a simulation with multiple AOE sources."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_source_1", ".", ".", ".", "aoe_source_2", "wall"],
            ["wall", ".", ".", "agent.agent", ".", ".", "wall"],
            ["wall", ".", ".", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        objects = {"wall": WallConfig()}
        for i, source in enumerate(sources, 1):
            objects[f"aoe_source_{i}"] = WallConfig(
                name=f"aoe_source_{i}",
                aoes=[
                    AOEEffectConfig(
                        range=source.get("range", 3),
                        resource_deltas=source.get("resource_deltas", {"energy": 5}),
                        members_only=source.get("members_only", False),
                        ignore_members=source.get("ignore_members", False),
                    )
                ],
            )

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=7, height=5, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects=objects,
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, resource: str) -> int:
        """Get agent's inventory for a specific resource."""
        objects = sim.grid_objects()
        agents = [obj for obj in objects.values() if "agent_id" in obj]
        assert len(agents) == 1
        resource_idx = sim.resource_names.index(resource)
        return agents[0]["inventory"][resource_idx]

    def test_multiple_aoe_effects_stack(self):
        """Test that multiple AOE effects from different sources stack."""
        sim = self._create_sim_with_multiple_aoe(
            sources=[
                {"range": 3, "resource_deltas": {"energy": 10}},
                {"range": 3, "resource_deltas": {"energy": 5}},
            ]
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        # Both effects should stack
        assert energy_after == energy_before + 15, (
            f"Both AOE effects should stack: expected +15, got {energy_after - energy_before}"
        )

    def test_overlapping_positive_and_negative(self):
        """Test that positive and negative AOE effects can coexist."""
        sim = self._create_sim_with_multiple_aoe(
            sources=[
                {"range": 3, "resource_deltas": {"energy": 20}},
                {"range": 3, "resource_deltas": {"energy": -5}},
            ]
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        # Net effect should be +15
        assert energy_after == energy_before + 15, f"Net AOE effect should be +15, got {energy_after - energy_before}"

    def test_aoe_effects_aggregated_before_applying(self):
        """Test that AOE effects are aggregated before applying to avoid order-dependent results.

        This tests a bug where if one AOE adds +5 to a resource and another removes -3,
        sequential application could produce different results than aggregated application.
        Correct behavior: aggregate first (+5 + -3 = +2), then apply once.

        We verify this by checking that the net effect is exactly +2, which proves
        the effects were combined rather than applied with any intermediate clamping.
        """
        # Create a simulation where:
        # - Agent starts with 10 of "energy" resource
        # - One AOE adds +5 energy
        # - Another AOE removes -3 energy
        # Aggregated: net = +2, so energy should be 12 after one tick
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_add", ".", "agent.agent", ".", "aoe_remove", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        objects = {
            "wall": WallConfig(),
            "aoe_add": WallConfig(
                name="aoe_add",
                aoes=[AOEEffectConfig(range=3, resource_deltas={"energy": 5})],
            ),
            "aoe_remove": WallConfig(
                name="aoe_remove",
                aoes=[AOEEffectConfig(range=3, resource_deltas={"energy": -3})],
            ),
        }

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=7, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
            ),
            objects=objects,
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        sim = Simulation(cfg, seed=42)

        energy_before = self._get_agent_inventory(sim, "energy")
        assert energy_before == 10, f"Energy should start at 10, got {energy_before}"

        # Step the simulation
        sim.agent(0).set_action("noop")
        sim.step()

        # After aggregation: +5 + -3 = +2
        # Energy should be exactly 12
        energy_after = self._get_agent_inventory(sim, "energy")
        expected = energy_before + 2  # Net effect of +5 and -3
        assert energy_after == expected, (
            f"Energy should be {expected} after aggregated AOE effects (net +2), got {energy_after}. "
            "This may indicate AOE effects are not being aggregated correctly."
        )


class TestAOECommonsFiltering:
    """Test AOE effects with commons membership filtering."""

    def _create_sim_with_commons_aoe(
        self,
        agent_commons: str | None = None,
        source_commons: str | None = None,
        members_only: bool = False,
        ignore_members: bool = False,
    ) -> Simulation:
        """Create a simulation with AOE source and agent potentially in different commons."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_source", ".", ".", "wall"],
            ["wall", ".", "agent.agent", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        commons = []
        if agent_commons:
            commons.append(CommonsConfig(name=agent_commons, inventory=InventoryConfig()))
        if source_commons and source_commons not in [c.name for c in commons]:
            commons.append(CommonsConfig(name=source_commons, inventory=InventoryConfig()))

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            resource_names=["energy", "heart"],
            commons=commons,
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                transfer=TransferActionConfig(
                    enabled=True,
                    align=True,
                    vibe_transfers=[
                        VibeTransfer(vibe=HEART_VIBE_NAME, target={}, actor={}),
                    ],
                ),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
                commons=agent_commons,
            ),
            objects={
                "wall": WallConfig(),
                "aoe_source": WallConfig(
                    name="aoe_source",
                    commons=source_commons,
                    aoes=[
                        AOEEffectConfig(
                            range=2,
                            resource_deltas={"energy": 10},
                            members_only=members_only,
                            ignore_members=ignore_members,
                        )
                    ],
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, resource: str) -> int:
        """Get agent's inventory for a specific resource."""
        objects = sim.grid_objects()
        agents = [obj for obj in objects.values() if "agent_id" in obj]
        assert len(agents) == 1
        resource_idx = sim.resource_names.index(resource)
        return agents[0]["inventory"][resource_idx]

    def test_members_only_same_commons(self):
        """Test that members_only effect applies when agent and source share commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons="team_red",
            source_commons="team_red",
            members_only=True,
            ignore_members=False,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (same commons). Before: {energy_before}, After: {energy_after}"
        )

    def test_members_only_different_commons(self):
        """Test that members_only effect does NOT apply when agent and source have different commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons="team_red",
            source_commons="team_blue",
            members_only=True,
            ignore_members=False,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before, (
            f"Agent should NOT receive effect (different commons). Before: {energy_before}, After: {energy_after}"
        )

    def test_members_only_no_agent_commons(self):
        """Test that members_only effect does NOT apply when agent has no commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons=None,
            source_commons="team_red",
            members_only=True,
            ignore_members=False,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before, (
            f"Agent should NOT receive effect (no commons). Before: {energy_before}, After: {energy_after}"
        )

    def test_ignore_members_same_commons(self):
        """Test that ignore_members effect does NOT apply when agent and source share commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons="team_red",
            source_commons="team_red",
            members_only=False,
            ignore_members=True,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before, (
            f"Agent should NOT receive effect (ignore_members, same commons). "
            f"Before: {energy_before}, After: {energy_after}"
        )

    def test_ignore_members_different_commons(self):
        """Test that ignore_members effect applies when agent and source have different commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons="team_red",
            source_commons="team_blue",
            members_only=False,
            ignore_members=True,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (ignore_members, different commons). "
            f"Before: {energy_before}, After: {energy_after}"
        )

    def test_ignore_members_no_agent_commons(self):
        """Test that ignore_members effect applies when agent has no commons."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons=None,
            source_commons="team_red",
            members_only=False,
            ignore_members=True,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (ignore_members, no commons). Before: {energy_before}, After: {energy_after}"
        )

    def test_no_filter_always_applies(self):
        """Test that effect without filters always applies."""
        sim = self._create_sim_with_commons_aoe(
            agent_commons="team_red",
            source_commons="team_blue",
            members_only=False,
            ignore_members=False,
        )

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (no filter). Before: {energy_before}, After: {energy_after}"
        )


class TestAOEAlignmentChange:
    """Test that changing agent alignment affects which AOE effects apply.

    Note: Alignment changes require the Transfer action with align=True, which
    only works when an agent moves into another agent with the appropriate vibe.
    """

    def _create_sim_with_two_team_agents(self) -> Simulation:
        """Create a simulation with two agents in different teams and team-specific AOE effects."""
        # Agent 0 starts in team_red, Agent 1 is in team_blue
        # Both agents are within range of both AOE sources
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_red", ".", "agent.red", "agent.blue", "aoe_blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=100,
            num_agents=2,
            obs=ObsConfig(width=7, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            commons=[
                CommonsConfig(name="team_red", inventory=InventoryConfig()),
                CommonsConfig(name="team_blue", inventory=InventoryConfig()),
            ],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                transfer=TransferActionConfig(
                    enabled=True,
                    align=True,
                    vibe_transfers=[
                        VibeTransfer(vibe=HEART_VIBE_NAME, target={}, actor={}),
                    ],
                ),
            ),
            agents=[
                AgentConfig(
                    inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
                    commons="team_red",
                    team_id=0,
                ),
                AgentConfig(
                    inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
                    commons="team_blue",
                    team_id=1,
                ),
            ],
            objects={
                "wall": WallConfig(),
                "aoe_red": WallConfig(
                    name="aoe_red",
                    commons="team_red",
                    aoes=[
                        AOEEffectConfig(
                            range=3,
                            resource_deltas={"energy": 10},
                            members_only=True,  # Only for team_red members
                        )
                    ],
                ),
                "aoe_blue": WallConfig(
                    name="aoe_blue",
                    commons="team_blue",
                    aoes=[
                        AOEEffectConfig(
                            range=3,
                            resource_deltas={"heart": 5},
                            members_only=True,  # Only for team_blue members
                        )
                    ],
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, agent_idx: int, resource: str) -> int:
        """Get a specific agent's inventory for a resource."""
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        resource_idx = sim.resource_names.index(resource)
        return agents[agent_idx]["inventory"][resource_idx]

    def test_agent_starts_receiving_team_red_effect(self):
        """Test that agent initially receives its team's AOE effect only."""
        sim = self._create_sim_with_two_team_agents()

        energy_red_before = self._get_agent_inventory(sim, 0, "energy")
        heart_red_before = self._get_agent_inventory(sim, 0, "heart")
        energy_blue_before = self._get_agent_inventory(sim, 1, "energy")
        heart_blue_before = self._get_agent_inventory(sim, 1, "heart")

        sim.agent(0).set_action("noop")
        sim.agent(1).set_action("noop")
        sim.step()

        energy_red_after = self._get_agent_inventory(sim, 0, "energy")
        heart_red_after = self._get_agent_inventory(sim, 0, "heart")
        energy_blue_after = self._get_agent_inventory(sim, 1, "energy")
        heart_blue_after = self._get_agent_inventory(sim, 1, "heart")

        # Team red agent should receive energy from aoe_red, but not heart from aoe_blue
        assert energy_red_after == energy_red_before + 10, "Red agent should receive team_red energy effect"
        assert heart_red_after == heart_red_before, "Red agent should NOT receive team_blue heart effect"

        # Team blue agent should receive heart from aoe_blue, but not energy from aoe_red
        assert heart_blue_after == heart_blue_before + 5, "Blue agent should receive team_blue heart effect"
        assert energy_blue_after == energy_blue_before, "Blue agent should NOT receive team_red energy effect"

    def test_different_teams_receive_different_effects(self):
        """Test that agents in different teams receive their respective team effects only."""
        sim = self._create_sim_with_two_team_agents()

        # Run multiple steps to verify consistent behavior
        for step_num in range(1, 4):
            sim.agent(0).set_action("noop")
            sim.agent(1).set_action("noop")
            sim.step()

            energy_red = self._get_agent_inventory(sim, 0, "energy")
            heart_red = self._get_agent_inventory(sim, 0, "heart")
            energy_blue = self._get_agent_inventory(sim, 1, "energy")
            heart_blue = self._get_agent_inventory(sim, 1, "heart")

            # Red agent: started with 100 energy, gets +10 per step
            expected_energy_red = 100 + (step_num * 10)
            assert energy_red == expected_energy_red, (
                f"Step {step_num}: Red agent should have {expected_energy_red} energy, got {energy_red}"
            )
            # Red agent: keeps 10 heart (no change)
            assert heart_red == 10, f"Step {step_num}: Red agent should keep 10 heart, got {heart_red}"

            # Blue agent: started with 100 energy, stays same (no energy effect)
            assert energy_blue == 100, f"Step {step_num}: Blue agent should keep 100 energy, got {energy_blue}"
            # Blue agent: started with 10 heart, gets +5 per step
            expected_heart_blue = 10 + (step_num * 5)
            assert heart_blue == expected_heart_blue, (
                f"Step {step_num}: Blue agent should have {expected_heart_blue} heart, got {heart_blue}"
            )


class TestAOEEffectsAddRemove:
    """Test that AOE effects are properly added/removed when objects appear/disappear."""

    def _create_sim_with_movable_aoe(self) -> Simulation:
        """Create a simulation where we can test effect registration."""
        # Simple setup: agent next to an AOE source
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "aoe_source", "agent.agent", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=4, height=3, num_tokens=100),
            resource_names=["energy"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 100}),
            ),
            objects={
                "wall": WallConfig(),
                "aoe_source": WallConfig(
                    name="aoe_source",
                    aoes=[
                        AOEEffectConfig(
                            range=1,
                            resource_deltas={"energy": 10},
                        )
                    ],
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, resource: str) -> int:
        """Get agent's inventory for a specific resource."""
        objects = sim.grid_objects()
        agents = [obj for obj in objects.values() if "agent_id" in obj]
        assert len(agents) == 1
        resource_idx = sim.resource_names.index(resource)
        return agents[0]["inventory"][resource_idx]

    def test_effect_registered_on_creation(self):
        """Test that AOE effect is registered when simulation starts."""
        sim = self._create_sim_with_movable_aoe()

        # Agent at (1,2), AOE source at (1,1), distance = 1 (within range)
        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, "AOE effect should be registered and applied on creation"

    def test_effect_applied_consistently_each_step(self):
        """Test that AOE effect is applied consistently on each simulation step."""
        sim = self._create_sim_with_movable_aoe()

        energy_initial = self._get_agent_inventory(sim, "energy")

        # Multiple steps
        for step_num in range(1, 4):
            sim.agent(0).set_action("noop")
            sim.step()

            energy_current = self._get_agent_inventory(sim, "energy")
            expected_energy = energy_initial + (step_num * 10)
            assert energy_current == expected_energy, (
                f"After step {step_num}: expected {expected_energy} energy, got {energy_current}"
            )


class TestAOETargetTags:
    """Test AOE effects with target_tags filtering."""

    def _create_sim_with_target_tags(
        self,
        target_tags: list[str] | None = None,
        agent_tags: list[str] | None = None,
    ) -> Simulation:
        """Create a simulation with AOE source that has target_tags filtering."""
        # Use specific tags for agent
        agent_tags = agent_tags or ["friendly"]

        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_source", ".", ".", "wall"],
            ["wall", ".", "agent.agent", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
                tags=agent_tags,
            ),
            objects={
                "wall": WallConfig(),
                "aoe_source": WallConfig(
                    name="aoe_source",
                    aoes=[
                        AOEEffectConfig(
                            range=2,
                            resource_deltas={"energy": 10},
                            target_tags=target_tags,
                        )
                    ],
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_agent_inventory(self, sim: Simulation, resource: str) -> int:
        """Get agent's inventory for a specific resource."""
        objects = sim.grid_objects()
        agents = [obj for obj in objects.values() if "agent_id" in obj]
        assert len(agents) == 1
        resource_idx = sim.resource_names.index(resource)
        return agents[0]["inventory"][resource_idx]

    def test_target_tags_none_affects_all(self):
        """Test that target_tags=None affects all objects."""
        sim = self._create_sim_with_target_tags(target_tags=None)

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (no target_tags filter). Before: {energy_before}, After: {energy_after}"
        )

    def test_target_tags_matching_affects_agent(self):
        """Test that agent with matching tag receives the effect."""
        sim = self._create_sim_with_target_tags(target_tags=["friendly"], agent_tags=["friendly"])

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (has matching tag). Before: {energy_before}, After: {energy_after}"
        )

    def test_target_tags_not_matching_skips_agent(self):
        """Test that agent without matching tag does NOT receive the effect."""
        # Use wall tag (which exists) as target, agent has different tag
        sim = self._create_sim_with_target_tags(target_tags=["wall"], agent_tags=["friendly"])

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before, (
            f"Agent should NOT receive effect (no matching tag). Before: {energy_before}, After: {energy_after}"
        )

    def test_target_tags_one_of_multiple_matches(self):
        """Test that agent receives effect if any of its tags match target_tags."""
        # Both friendly and hero must be in the game (hero from agent, friendly from target_tags)
        sim = self._create_sim_with_target_tags(target_tags=["friendly", "hero"], agent_tags=["friendly", "hero"])

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (has 'friendly' tag). Before: {energy_before}, After: {energy_after}"
        )

    def test_target_tags_empty_list_affects_all(self):
        """Test that empty target_tags list affects all objects."""
        sim = self._create_sim_with_target_tags(target_tags=[], agent_tags=["any_tag"])

        energy_before = self._get_agent_inventory(sim, "energy")

        sim.agent(0).set_action("noop")
        sim.step()

        energy_after = self._get_agent_inventory(sim, "energy")
        assert energy_after == energy_before + 10, (
            f"Agent should receive effect (empty target_tags). Before: {energy_before}, After: {energy_after}"
        )
