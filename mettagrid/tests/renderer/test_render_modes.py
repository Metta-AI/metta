"""Tests for different render modes and CI/environment detection."""

import os
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from metta.common.util.test_fixture import get_cfg
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum import SingleTaskCurriculum


class TestRenderModes:
    """Test different render modes and their behavior in various environments."""

    def _create_env(self, render_mode=None):
        """Helper to create a test environment."""
        cfg = get_cfg("benchmark")
        cfg.game.num_agents = 1
        cfg.game.max_steps = 5
        cfg.game.map_builder = OmegaConf.create(
            {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 5,
                "height": 5,
                "agents": 1,
                "border_width": 1,
                "objects": {},
            }
        )
        curriculum = SingleTaskCurriculum("test", cfg)
        return MettaGridEnv(curriculum, render_mode=render_mode)

    def test_no_render_mode(self):
        """Test that render returns None when no render_mode is set."""
        env = self._create_env(render_mode=None)
        obs, info = env.reset()

        # Should return None when no render mode and raylib available
        result = env.render()
        # Result depends on whether raylib is available
        assert result is None or isinstance(result, type(None))

        env.close()

    def test_human_render_mode(self):
        """Test NetHack-style human rendering."""
        env = self._create_env(render_mode="human")
        obs, info = env.reset()

        result = env.render()
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Check for NetHack-style characters
        assert any(c in result for c in ["#", ".", "0"])  # walls, empty, agent

        env.close()

    def test_miniscope_render_mode(self):
        """Test emoji-based miniscope rendering."""
        env = self._create_env(render_mode="miniscope")
        obs, info = env.reset()

        result = env.render()
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain emoji characters
        # Check for some common emoji bytes
        assert any(ord(c) > 127 for c in result)  # Contains non-ASCII

        env.close()

    @patch.dict(os.environ, {"CI": "true"})
    def test_ci_environment_detection(self):
        """Test that CI environment is properly detected."""
        env = self._create_env(render_mode=None)

        # Should have detected CI environment
        assert env._is_ci_environment is True
        assert env._raylib_available is False

        # Render should return None in CI without text renderer
        assert env.render() is None

        env.close()

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"})
    def test_github_actions_detection(self):
        """Test that GitHub Actions environment is detected."""
        env = self._create_env(render_mode=None)

        assert env._is_ci_environment is True
        assert env._raylib_available is False

        env.close()

    @patch("os.path.exists")
    def test_docker_detection(self, mock_exists):
        """Test that Docker environment is detected."""
        # Mock /.dockerenv exists
        mock_exists.return_value = True

        env = self._create_env(render_mode=None)

        assert env._is_ci_environment is True
        assert env._raylib_available is False

        env.close()

    def test_render_consistency(self):
        """Test that repeated renders produce consistent output."""
        env = self._create_env(render_mode="human")
        obs, info = env.reset()

        # Get two renders without stepping
        render1 = env.render()
        render2 = env.render()

        # Should be identical if no actions taken
        assert render1 == render2

        # Take an action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Render should change after action
        render3 = env.render()
        assert render3 != render1 or done  # May be same if episode ended

        env.close()

    def test_render_performance(self):
        """Test that render method is fast."""
        import time

        env = self._create_env(render_mode="human")
        obs, info = env.reset()

        # Warm up
        env.render()

        # Time multiple renders
        start = time.perf_counter()
        n_renders = 100
        for _ in range(n_renders):
            env.render()
        elapsed = time.perf_counter() - start

        # Should be very fast (< 1ms per render for text)
        avg_time = elapsed / n_renders
        assert avg_time < 0.001  # Less than 1ms per render

        env.close()


class TestRaylibIntegration:
    """Test raylib-specific rendering behavior."""

    @pytest.mark.skipif("CI" in os.environ or "GITHUB_ACTIONS" in os.environ, reason="Raylib tests skipped in CI")
    def test_raylib_initialization(self):
        """Test raylib renderer initialization (only runs locally)."""
        from metta.mettagrid import mettagrid_env

        if mettagrid_env.RAYLIB_AVAILABLE:
            cfg = get_cfg("benchmark")
            cfg.game.num_agents = 1
            curriculum = SingleTaskCurriculum("test", cfg)

            # Create env without specifying render_mode
            env = MettaGridEnv(curriculum, render_mode=None)

            # Should have raylib available
            assert env._raylib_available is True

            env.close()

    def test_raylib_fallback(self):
        """Test behavior when raylib is not available."""
        # Mock raylib as unavailable
        with patch("metta.mettagrid.mettagrid_env.RAYLIB_AVAILABLE", False):
            cfg = get_cfg("benchmark")
            cfg.game.num_agents = 1
            curriculum = SingleTaskCurriculum("test", cfg)

            env = MettaGridEnv(curriculum, render_mode=None)

            # Should not have raylib available
            assert env._raylib_available is False

            # Render should return None
            obs, info = env.reset()
            assert env.render() is None

            env.close()
