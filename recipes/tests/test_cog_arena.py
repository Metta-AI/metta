"""Tests for the cog_arena recipe."""

from __future__ import annotations

import pytest

from recipes.experiment import cog_arena


class TestCogArenaVibes:
    """Tests for vibe configuration."""

    def test_vibes_count(self) -> None:
        """Verify we have the expected 6 vibes."""
        assert len(cog_arena.VIBES) == 6

    def test_vibe_names(self) -> None:
        """Verify the expected vibe names are present."""
        vibe_names = {vibe.name for vibe in cog_arena.VIBES}
        expected = {"default", "weapon", "shield", "battery", "gear", "heart"}
        assert vibe_names == expected


class TestCogArenaResources:
    """Tests for resource configuration."""

    def test_resources_count(self) -> None:
        """Verify we have the expected 11 resources."""
        assert len(cog_arena.RESOURCES) == 11

    def test_resource_names(self) -> None:
        """Verify the expected resource names are present."""
        expected = {
            "energy",
            "carbon",
            "oxygen",
            "germanium",
            "silicon",
            "heart",
            "weapon",
            "shield",
            "battery",
            "gear",
            "damage",
        }
        assert set(cog_arena.RESOURCES) == expected


class TestCogArenaEnvConfig:
    """Tests for environment configuration."""

    def test_make_env_default(self) -> None:
        """Verify make_env creates a valid environment with defaults."""
        env = cog_arena.make_env()
        assert env is not None
        assert env.game.num_agents == 10

    def test_make_env_custom_agents(self) -> None:
        """Verify make_env respects custom agent count."""
        env = cog_arena.make_env(num_agents=20)
        assert env.game.num_agents == 20

    def test_env_has_damage_config(self) -> None:
        """Verify the environment has damage configuration."""
        env = cog_arena.make_env()
        assert env.game.agent.damage is not None
        assert "damage" in env.game.agent.damage.threshold
        assert env.game.agent.damage.threshold["damage"] == 200

    def test_env_has_transfer_action(self) -> None:
        """Verify the environment has transfer action configured."""
        env = cog_arena.make_env()
        transfer = env.game.actions.transfer
        assert len(transfer.vibe_transfers) == 3
        assert "battery" in transfer.vibes
        assert "heart" in transfer.vibes
        assert "gear" in transfer.vibes

    def test_env_has_attack_vibes(self) -> None:
        """Verify the attack action has vibe triggers configured."""
        env = cog_arena.make_env()
        attack = env.game.actions.attack
        assert "weapon" in attack.vibes

    def test_env_has_resource_limit_modifiers(self) -> None:
        """Verify resource limits have modifiers configured."""
        env = cog_arena.make_env()
        energy_limit = env.game.agent.resource_limits.get("energy")
        assert energy_limit is not None
        assert "battery" in energy_limit.modifiers
        assert energy_limit.modifiers["battery"] == 25

    def test_env_has_inventory_order(self) -> None:
        """Verify inventory order is configured for modifier dependencies."""
        env = cog_arena.make_env()
        assert env.game.agent.inventory_order == ["gear", "battery", "weapon", "shield"]

    def test_env_has_inventory_regen(self) -> None:
        """Verify vibe-dependent inventory regeneration is configured."""
        env = cog_arena.make_env()
        regen = env.game.agent.inventory_regen_amounts
        assert "default" in regen
        assert "energy" in regen["default"]
        assert "damage" in regen["default"]


class TestCogArenaCurriculum:
    """Tests for curriculum configuration."""

    def test_make_curriculum_default(self) -> None:
        """Verify make_curriculum creates a valid curriculum."""
        curriculum = cog_arena.make_curriculum()
        assert curriculum is not None

    def test_curriculum_has_buckets(self) -> None:
        """Verify curriculum has task buckets configured."""
        curriculum = cog_arena.make_curriculum()
        # The curriculum should have multiple task configurations
        assert curriculum.task_manager is not None


class TestCogArenaSimulations:
    """Tests for simulation configuration."""

    def test_simulations_default(self) -> None:
        """Verify simulations creates valid configs."""
        sims = cog_arena.simulations()
        assert len(sims) == 1
        assert sims[0].suite == "cog_arena"
        assert sims[0].name == "basic"


class TestCogArenaTools:
    """Tests for tool configurations."""

    def test_train_tool(self) -> None:
        """Verify train tool can be created."""
        tool = cog_arena.train()
        assert tool is not None

    def test_evaluate_tool(self) -> None:
        """Verify evaluate tool can be created."""
        tool = cog_arena.evaluate()
        assert tool is not None

    def test_play_tool(self) -> None:
        """Verify play tool can be created."""
        tool = cog_arena.play()
        assert tool is not None

    def test_replay_tool(self) -> None:
        """Verify replay tool can be created."""
        tool = cog_arena.replay()
        assert tool is not None


class TestCogAssemblerConfig:
    """Tests for the CogAssemblerConfig."""

    def test_assembler_has_protocols(self) -> None:
        """Verify assembler has expected protocols."""
        config = cog_arena.CogAssemblerConfig()
        assembler = config.station_cfg()
        assert len(assembler.protocols) == 5

    def test_assembler_protocol_vibes(self) -> None:
        """Verify assembler protocols have correct vibes."""
        config = cog_arena.CogAssemblerConfig()
        assembler = config.station_cfg()
        protocol_vibes = {tuple(p.vibes) for p in assembler.protocols}
        expected_vibes = {("heart",), ("weapon",), ("shield",), ("battery",), ("gear",)}
        assert protocol_vibes == expected_vibes

    def test_assembler_has_agent_cooldown(self) -> None:
        """Verify assembler has per-agent cooldown configured."""
        config = cog_arena.CogAssemblerConfig()
        assembler = config.station_cfg()
        assert assembler.agent_cooldown == 10

    def test_assembler_has_chest_search_distance(self) -> None:
        """Verify assembler can search nearby chests."""
        config = cog_arena.CogAssemblerConfig()
        assembler = config.station_cfg()
        assert assembler.chest_search_distance == 10

    def test_assembler_protocol_inflation(self) -> None:
        """Verify assembler protocols have inflation configured."""
        config = cog_arena.CogAssemblerConfig()
        assembler = config.station_cfg()
        for protocol in assembler.protocols:
            assert protocol.inflation > 0
            assert protocol.sigmoid > 0
