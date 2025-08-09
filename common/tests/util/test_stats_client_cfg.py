"""Tests for metta.common.util.stats_client_cfg module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from metta.common.util.stats_client_cfg import get_machine_token, get_stats_client, get_stats_client_direct


class TestGetMachineToken:
    """Test cases for the get_machine_token function."""

    def test_get_machine_token_from_yaml_file(self):
        """Test getting machine token from YAML tokens file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            tokens_file.parent.mkdir(parents=True)

            # Create tokens file
            tokens = {
                "https://api.example.com": "token123",
                "https://api.test.com": "token456"
            }

            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Test specific server URI
                token = get_machine_token("https://api.example.com")
                assert token == "token123"

                # Test different server URI
                token = get_machine_token("https://api.test.com")
                assert token == "token456"

                # Test non-existent server URI
                token = get_machine_token("https://api.notfound.com")
                assert token is None

    def test_get_machine_token_strips_whitespace(self):
        """Test that machine token is stripped of whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            tokens_file.parent.mkdir(parents=True)

            # Create tokens file with whitespace
            tokens = {
                "https://api.example.com": "  token123  \n",
            }

            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token == "token123"

    def test_get_machine_token_handles_empty_yaml(self):
        """Test handling of empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            tokens_file.parent.mkdir(parents=True)

            # Create empty YAML file
            with open(tokens_file, "w") as f:
                f.write("")

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

    def test_get_machine_token_handles_non_dict_yaml(self):
        """Test handling of YAML file that doesn't contain a dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            tokens_file.parent.mkdir(parents=True)

            # Create YAML file with list instead of dict
            with open(tokens_file, "w") as f:
                yaml.dump(["token1", "token2"], f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

    def test_get_machine_token_legacy_file_for_none_uri(self):
        """Test fallback to legacy token file when URI is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_file = Path(temp_dir) / ".metta" / "observatory_token"
            legacy_file.parent.mkdir(parents=True)

            # Create legacy token file
            with open(legacy_file, "w") as f:
                f.write("legacy_token123")

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token(None)
                assert token == "legacy_token123"

    def test_get_machine_token_legacy_file_for_prod_uri(self):
        """Test fallback to legacy token file for production URIs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_file = Path(temp_dir) / ".metta" / "observatory_token"
            legacy_file.parent.mkdir(parents=True)

            # Create legacy token file
            with open(legacy_file, "w") as f:
                f.write("prod_token456")

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Test with specific production URI
                token = get_machine_token("https://observatory.softmax-research.net/api")
                assert token == "prod_token456"

                # Test with PROD_STATS_SERVER_URI
                with patch("metta.common.util.stats_client_cfg.PROD_STATS_SERVER_URI", "https://prod.example.com"):
                    token = get_machine_token("https://prod.example.com")
                    assert token == "prod_token456"

    def test_get_machine_token_legacy_strips_whitespace(self):
        """Test that legacy token is stripped of whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_file = Path(temp_dir) / ".metta" / "observatory_token"
            legacy_file.parent.mkdir(parents=True)

            # Create legacy token file with whitespace
            with open(legacy_file, "w") as f:
                f.write("  legacy_token  \n")

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token(None)
                assert token == "legacy_token"

    def test_get_machine_token_returns_none_for_invalid_tokens(self):
        """Test that invalid tokens return None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test empty token
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            tokens_file.parent.mkdir(parents=True)

            tokens = {"https://api.example.com": ""}
            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

            # Test "none" token (case insensitive)
            tokens = {"https://api.example.com": "None"}
            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

            # Test "NONE" token
            tokens = {"https://api.example.com": "NONE"}
            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

    def test_get_machine_token_no_files_exist(self):
        """Test behavior when no token files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory, no token files
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                token = get_machine_token("https://api.example.com")
                assert token is None

                token = get_machine_token(None)
                assert token is None

    def test_get_machine_token_yaml_takes_precedence(self):
        """Test that YAML file takes precedence over legacy file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create both files
            tokens_file = Path(temp_dir) / ".metta" / "observatory_tokens.yaml"
            legacy_file = Path(temp_dir) / ".metta" / "observatory_token"
            tokens_file.parent.mkdir(parents=True)

            # YAML file
            tokens = {"https://api.example.com": "yaml_token"}
            with open(tokens_file, "w") as f:
                yaml.dump(tokens, f)

            # Legacy file
            with open(legacy_file, "w") as f:
                f.write("legacy_token")

            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Should prefer YAML file
                token = get_machine_token("https://api.example.com")
                assert token == "yaml_token"

    def test_get_machine_token_unknown_uri_returns_none(self):
        """Test that unknown URI with no fallback returns None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                # Unknown URI, no files exist
                token = get_machine_token("https://unknown.example.com")
                assert token is None


class TestGetStatsClientDirect:
    """Test cases for the get_stats_client_direct function."""

    def test_get_stats_client_direct_with_none_uri(self):
        """Test that None URI returns None and logs warning."""
        logger = Mock(spec=logging.Logger)

        result = get_stats_client_direct(None, logger)

        assert result is None
        logger.warning.assert_called_once_with("No stats server URI provided, running without stats collection")

    @patch("metta.common.util.stats_client_cfg.get_machine_token")
    def test_get_stats_client_direct_with_none_token(self, mock_get_token):
        """Test that None token returns None and logs warning."""
        mock_get_token.return_value = None
        logger = Mock(spec=logging.Logger)

        result = get_stats_client_direct("https://api.example.com", logger)

        assert result is None
        mock_get_token.assert_called_once_with("https://api.example.com")
        logger.warning.assert_called_once_with("No machine token provided, running without stats collection")

    @patch("metta.common.util.stats_client_cfg.get_machine_token")
    @patch("metta.common.util.stats_client_cfg.StatsClient")
    def test_get_stats_client_direct_success(self, mock_stats_client, mock_get_token):
        """Test successful creation of StatsClient."""
        mock_get_token.return_value = "valid_token"
        mock_client_instance = Mock()
        mock_stats_client.return_value = mock_client_instance
        logger = Mock(spec=logging.Logger)

        result = get_stats_client_direct("https://api.example.com", logger)

        assert result == mock_client_instance
        mock_get_token.assert_called_once_with("https://api.example.com")
        mock_stats_client.assert_called_once_with(
            backend_url="https://api.example.com",
            machine_token="valid_token"
        )
        logger.info.assert_called_once_with("Using stats client at https://api.example.com")


class TestGetStatsClient:
    """Test cases for the get_stats_client function."""

    def test_get_stats_client_with_non_dictconfig(self):
        """Test that non-DictConfig returns None."""
        logger = Mock(spec=logging.Logger)

        # Test with ListConfig
        list_cfg = OmegaConf.create(["item1", "item2"])
        result = get_stats_client(list_cfg, logger)
        assert result is None

        # Test with None
        result = get_stats_client(None, logger)
        assert result is None

    @patch("metta.common.util.stats_client_cfg.get_stats_client_direct")
    def test_get_stats_client_with_dictconfig(self, mock_get_direct):
        """Test that DictConfig is handled correctly."""
        mock_get_direct.return_value = Mock()
        logger = Mock(spec=logging.Logger)

        # Test with stats_server_uri present
        cfg = OmegaConf.create({"stats_server_uri": "https://api.example.com"})
        result = get_stats_client(cfg, logger)

        mock_get_direct.assert_called_once_with("https://api.example.com", logger)
        assert result == mock_get_direct.return_value

    @patch("metta.common.util.stats_client_cfg.get_stats_client_direct")
    def test_get_stats_client_with_missing_uri(self, mock_get_direct):
        """Test that missing stats_server_uri defaults to None."""
        mock_get_direct.return_value = None
        logger = Mock(spec=logging.Logger)

        # Test with empty DictConfig
        cfg = OmegaConf.create({})
        result = get_stats_client(cfg, logger)

        mock_get_direct.assert_called_once_with(None, logger)
        assert result is None

    @patch("metta.common.util.stats_client_cfg.get_stats_client_direct")
    def test_get_stats_client_with_none_uri(self, mock_get_direct):
        """Test that explicit None URI is handled."""
        mock_get_direct.return_value = None
        logger = Mock(spec=logging.Logger)

        # Test with explicit None
        cfg = OmegaConf.create({"stats_server_uri": None})
        result = get_stats_client(cfg, logger)

        mock_get_direct.assert_called_once_with(None, logger)
        assert result is None

    @patch("metta.common.util.stats_client_cfg.get_stats_client_direct")
    def test_get_stats_client_integration(self, mock_get_direct):
        """Test integration with realistic config."""
        mock_client = Mock()
        mock_get_direct.return_value = mock_client
        logger = Mock(spec=logging.Logger)

        # Test realistic config
        cfg = OmegaConf.create({
            "stats_server_uri": "https://observatory.example.com/api",
            "other_setting": "value",
            "nested": {"config": "test"}
        })

        result = get_stats_client(cfg, logger)

        mock_get_direct.assert_called_once_with("https://observatory.example.com/api", logger)
        assert result == mock_client
