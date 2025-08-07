"""Tests for metta.common.util.mettagrid_cfgs module."""

import os
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.common.util.mettagrid_cfgs import (
    MettagridCfgFile,
    MettagridCfgFileMetadata,
)


class TestMettagridCfgFileMetadata:
    """Test cases for MettagridCfgFileMetadata class."""

    def test_metadata_init(self):
        """Test MettagridCfgFileMetadata initialization."""
        metadata = MettagridCfgFileMetadata(path="test/config.yaml", kind="env")
        assert metadata.path == "test/config.yaml"
        assert metadata.kind == "env"

    def test_from_path_map_config(self):
        """Test from_path classification for map configs."""
        result = MettagridCfgFileMetadata.from_path("game/map_builder/test_map.yaml")
        assert result.path == "game/map_builder/test_map.yaml"
        assert result.kind == "map"

    def test_from_path_curriculum_config(self):
        """Test from_path classification for curriculum configs."""
        result = MettagridCfgFileMetadata.from_path("curriculum/basic_training.yaml")
        assert result.path == "curriculum/basic_training.yaml"
        assert result.kind == "curriculum"

    def test_from_path_env_config(self):
        """Test from_path classification for env configs."""
        result = MettagridCfgFileMetadata.from_path("environments/standard.yaml")
        assert result.path == "environments/standard.yaml"
        assert result.kind == "env"

    def test_from_path_unknown_config(self):
        """Test from_path classification for configs that default to env."""
        result = MettagridCfgFileMetadata.from_path("other/weird_config.yaml")
        assert result.path == "other/weird_config.yaml"
        # Per the heuristic logic, anything not matching specific patterns defaults to "env"
        assert result.kind == "env"

    @patch('os.walk')
    def test_get_all_configs(self, mock_walk):
        """Test get_all method that scans config directories."""
        # Mock directory structure
        mock_walk.return_value = [
            ("configs/env/mettagrid", [], ["env1.yaml", "env2.yaml", "readme.txt"]),
            ("configs/env/mettagrid/game/map_builder", [], ["map1.yaml", "map2.yaml"]),
            ("configs/env/mettagrid/curriculum", [], ["basic.yaml", "advanced.yaml"]),
        ]

        result = MettagridCfgFileMetadata.get_all()

        # Should group configs by kind
        assert "env" in result
        assert "map" in result
        assert "curriculum" in result

        # Should filter out non-yaml files
        env_paths = [meta.path for meta in result["env"]]
        assert "env1.yaml" in env_paths
        assert "env2.yaml" in env_paths
        assert "readme.txt" not in str(env_paths)  # Not a yaml file

        # Should correctly classify map configs
        map_paths = [meta.path for meta in result["map"]]
        assert any("map1.yaml" in path for path in map_paths)
        assert any("map2.yaml" in path for path in map_paths)

        # Should correctly classify curriculum configs
        curriculum_paths = [meta.path for meta in result["curriculum"]]
        assert any("basic.yaml" in path for path in curriculum_paths)
        assert any("advanced.yaml" in path for path in curriculum_paths)

    @patch('hydra.initialize')
    @patch('metta.common.util.mettagrid_cfgs.config_from_path')
    def test_get_cfg(self, mock_config_from_path, mock_hydra_init):
        """Test get_cfg method that loads config using Hydra."""
        # Mock config loading
        mock_cfg = OmegaConf.create({"test": "value"})
        mock_config_from_path.return_value = mock_cfg
        mock_hydra_context = Mock()
        mock_hydra_init.return_value.__enter__ = Mock(return_value=mock_hydra_context)
        mock_hydra_init.return_value.__exit__ = Mock(return_value=None)

        metadata = MettagridCfgFileMetadata(path="test/config.yaml", kind="env")
        result = metadata.get_cfg()

        # Should initialize Hydra with correct path
        mock_hydra_init.assert_called_once_with(
            config_path="../../../../../configs",
            version_base=None
        )

        # Should load config from correct path
        mock_config_from_path.assert_called_once_with("env/mettagrid/test/config.yaml")

        # Should return MettagridCfgFile
        assert isinstance(result, MettagridCfgFile)
        assert result.metadata == metadata
        assert result.cfg == mock_cfg

    @patch('hydra.initialize')
    @patch('metta.common.util.mettagrid_cfgs.config_from_path')
    def test_get_cfg_invalid_type(self, mock_config_from_path, mock_hydra_init):
        """Test get_cfg with invalid config type."""
        # Mock config loading returning non-DictConfig
        mock_config_from_path.return_value = "not a dict config"
        mock_hydra_context = Mock()
        mock_hydra_init.return_value.__enter__ = Mock(return_value=mock_hydra_context)
        mock_hydra_init.return_value.__exit__ = Mock(return_value=None)

        metadata = MettagridCfgFileMetadata(path="test/config.yaml", kind="env")

        with pytest.raises(ValueError, match="Invalid config type"):
            metadata.get_cfg()

    @patch('os.getcwd')
    def test_absolute_path(self, mock_getcwd):
        """Test absolute_path method."""
        mock_getcwd.return_value = "/current/dir"

        metadata = MettagridCfgFileMetadata(path="test/config.yaml", kind="env")
        result = metadata.absolute_path()

        expected = "/current/dir/configs/env/mettagrid/test/config.yaml"
        assert result == expected

    @patch('os.getcwd')
    def test_to_dict(self, mock_getcwd):
        """Test to_dict method."""
        mock_getcwd.return_value = "/current/dir"

        metadata = MettagridCfgFileMetadata(path="test/config.yaml", kind="env")
        result = metadata.to_dict()

        expected = {
            "absolute_path": "/current/dir/configs/env/mettagrid/test/config.yaml",
            "path": "test/config.yaml",
            "kind": "env"
        }
        assert result == expected


class TestMettagridCfgFile:
    """Test cases for MettagridCfgFile class."""

    def test_cfg_file_init(self):
        """Test MettagridCfgFile initialization."""
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="env")
        cfg = OmegaConf.create({"test": "value"})

        cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)
        assert cfg_file.metadata == metadata
        assert cfg_file.cfg == cfg

    def test_to_dict(self):
        """Test to_dict method."""
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="env")
        cfg = OmegaConf.create({"test": "value", "nested": {"key": "val"}})

        with patch.object(metadata, 'to_dict') as mock_metadata_to_dict:
            mock_metadata_to_dict.return_value = {"metadata": "dict"}

            cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)
            result = cfg_file.to_dict()

            expected = {
                "metadata": {"metadata": "dict"},
                "cfg": {"test": "value", "nested": {"key": "val"}}
            }
            assert result == expected

    @patch.object(MettagridCfgFileMetadata, 'get_cfg')
    def test_from_path(self, mock_get_cfg):
        """Test from_path class method."""
        # Mock the chain of calls
        mock_cfg_file = Mock()
        mock_get_cfg.return_value = mock_cfg_file

        with patch.object(MettagridCfgFileMetadata, 'from_path') as mock_from_path:
            mock_metadata = Mock()
            mock_from_path.return_value = mock_metadata
            mock_metadata.get_cfg.return_value = mock_cfg_file

            result = MettagridCfgFile.from_path("test/path.yaml")

            mock_from_path.assert_called_once_with("test/path.yaml")
            mock_metadata.get_cfg.assert_called_once()
            assert result == mock_cfg_file

    def test_get_map_cfg_map_kind(self):
        """Test get_map_cfg with map kind config."""
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="map")
        cfg = OmegaConf.create({"map_config": "data"})

        cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)
        result = cfg_file.get_map_cfg()

        assert result == cfg

    def test_get_map_cfg_env_kind(self):
        """Test get_map_cfg with env kind config."""
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="env")
        map_cfg = OmegaConf.create({"map_data": "info"})
        cfg = OmegaConf.create({"game": {"map_builder": map_cfg}})

        cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)
        result = cfg_file.get_map_cfg()

        assert result == map_cfg

    def test_get_map_cfg_invalid_kind(self):
        """Test get_map_cfg with invalid config kind."""
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="curriculum")
        cfg = OmegaConf.create({"some": "data"})

        cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)

        with pytest.raises(ValueError, match="Config test.yaml is not a map or env"):
            cfg_file.get_map_cfg()


class TestMettagridCfgIntegration:
    """Integration tests for mettagrid config functionality."""

    @patch('os.walk')
    @patch('os.getcwd')
    def test_full_workflow(self, mock_getcwd, mock_walk):
        """Test a complete workflow from discovery to config loading."""
        mock_getcwd.return_value = "/project"
        mock_walk.return_value = [
            ("configs/env/mettagrid", [], ["basic_env.yaml"]),
            ("configs/env/mettagrid/game/map_builder", [], ["test_map.yaml"]),
        ]

        # Test discovery
        all_configs = MettagridCfgFileMetadata.get_all()

        assert "env" in all_configs
        assert "map" in all_configs

        env_config = all_configs["env"][0]
        map_config = all_configs["map"][0]

        # Test metadata properties
        assert env_config.kind == "env"
        assert map_config.kind == "map"

        # Test absolute path generation
        env_abs_path = env_config.absolute_path()
        assert env_abs_path == "/project/configs/env/mettagrid/basic_env.yaml"

        map_abs_path = map_config.absolute_path()
        assert "/project/configs/env/mettagrid" in map_abs_path
        assert "test_map.yaml" in map_abs_path

    def test_config_kind_classification_edge_cases(self):
        """Test edge cases in config kind classification."""
        test_cases = [
            ("game/map_builder/nested/deep_map.yaml", "map"),
            ("curriculum/advanced/training.yaml", "curriculum"),
            ("environments/simple.yaml", "env"),
            ("game/other/config.yaml", "env"),  # Not map_builder
            ("something/completely/different.yaml", "env"),  # Default
            ("", "env"),  # Empty path
        ]

        for path, expected_kind in test_cases:
            result = MettagridCfgFileMetadata.from_path(path)
            assert result.kind == expected_kind, f"Path {path} should be {expected_kind}, got {result.kind}"

    @patch('os.walk')
    def test_get_all_empty_directory(self, mock_walk):
        """Test get_all with empty directory structure."""
        mock_walk.return_value = []

        result = MettagridCfgFileMetadata.get_all()

        assert isinstance(result, dict)
        assert len(result) == 0

    @patch('os.walk')
    def test_get_all_mixed_file_types(self, mock_walk):
        """Test get_all filters non-yaml files correctly."""
        mock_walk.return_value = [
            ("configs/env/mettagrid", [], [
                "config.yaml",
                "readme.md",
                "data.json",
                "script.py",
                "another.yaml",
                ".hidden.yaml"
            ]),
        ]

        result = MettagridCfgFileMetadata.get_all()

        # Should only include .yaml files
        assert "env" in result
        yaml_files = [meta.path for meta in result["env"]]
        assert "config.yaml" in yaml_files
        assert "another.yaml" in yaml_files
        assert ".hidden.yaml" in yaml_files  # Hidden files are still included if .yaml

        # Should exclude non-yaml files
        all_paths = str(yaml_files)
        assert "readme.md" not in all_paths
        assert "data.json" not in all_paths
        assert "script.py" not in all_paths

    def test_cfg_file_type_annotations(self):
        """Test that MettagridCfgFile.AsDict type annotation is correct."""
        # This test ensures the TypedDict is properly defined
        from metta.common.util.mettagrid_cfgs import MettagridCfgFile

        # Check that AsDict is a class attribute
        assert hasattr(MettagridCfgFile, 'AsDict')

        # Create a proper instance
        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="env")
        cfg = OmegaConf.create({"test": "value"})
        cfg_file = MettagridCfgFile(metadata=metadata, cfg=cfg)

        # Test to_dict returns proper structure
        result = cfg_file.to_dict()
        assert "metadata" in result
        assert "cfg" in result
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["cfg"], dict)

    @patch('hydra.initialize')
    @patch('metta.common.util.mettagrid_cfgs.config_from_path')
    def test_hydra_context_management(self, mock_config_from_path, mock_hydra_init):
        """Test that Hydra context is properly managed."""
        mock_cfg = OmegaConf.create({"test": "value"})
        mock_config_from_path.return_value = mock_cfg

        # Mock context manager properly
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=None)
        mock_hydra_init.return_value = mock_context

        metadata = MettagridCfgFileMetadata(path="test.yaml", kind="env")

        # Call get_cfg
        result = metadata.get_cfg()

        # Verify context manager was used
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

        # Verify result
        assert isinstance(result, MettagridCfgFile)
        assert result.cfg == mock_cfg
