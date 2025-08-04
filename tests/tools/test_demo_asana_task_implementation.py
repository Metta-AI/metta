#!/usr/bin/env python3
"""
Tests for the demo Asana task implementation (mettagrid config validator).
"""

import os
import tempfile
import pytest
from tools.demo_asana_task_implementation import MettagridConfigValidator, validate_config_file


class TestMettagridConfigValidator:
    """Test the MettagridConfigValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return MettagridConfigValidator()
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration."""
        return {
            "defaults": [
                "/env/mettagrid/mettagrid@",
                "_self_"
            ],
            "game": {
                "num_agents": 10,
                "map_builder": {
                    "_target_": "metta.map.mapgen.MapGen",
                    "width": 25,
                    "height": 25,
                    "instances": 2,
                    "root": {
                        "type": "metta.map.scenes.random.Random",
                        "params": {
                            "agents": 5,
                            "objects": {
                                "wall": 10,
                                "mine": 5
                            }
                        }
                    }
                },
                "objects": {
                    "altar": {
                        "cooldown": 1,
                        "initial_resource_count": 1
                    }
                }
            }
        }
    
    def test_validate_required_fields(self, validator):
        """Test validation of required fields."""
        # Missing defaults
        config = {"game": {}}
        validator._validate_required_fields(config)
        assert len(validator.errors) == 1
        assert "defaults" in validator.errors[0]
        
        # Missing game
        validator.errors = []
        config = {"defaults": []}
        validator._validate_required_fields(config)
        assert len(validator.errors) == 1
        assert "game" in validator.errors[0]
        
        # All required fields present
        validator.errors = []
        config = {"defaults": [], "game": {}}
        validator._validate_required_fields(config)
        assert len(validator.errors) == 0
    
    def test_validate_num_agents(self, validator):
        """Test validation of num_agents."""
        # Valid num_agents
        game_config = {"num_agents": 10}
        validator._validate_game_config(game_config)
        assert len(validator.errors) == 0
        
        # Invalid type
        validator.errors = []
        game_config = {"num_agents": "ten"}
        validator._validate_game_config(game_config)
        assert len(validator.errors) == 1
        assert "positive integer" in validator.errors[0]
        
        # Negative value
        validator.errors = []
        game_config = {"num_agents": -5}
        validator._validate_game_config(game_config)
        assert len(validator.errors) == 1
        assert "positive integer" in validator.errors[0]
        
        # Very high value (warning)
        validator.errors = []
        validator.warnings = []
        game_config = {"num_agents": 150}
        validator._validate_game_config(game_config)
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 1
        assert "very high" in validator.warnings[0]
    
    def test_validate_map_dimensions(self, validator):
        """Test validation of map dimensions."""
        # Valid dimensions
        map_config = {"width": 25, "height": 30}
        validator._validate_map_builder(map_config)
        assert len(validator.errors) == 0
        
        # Invalid width
        validator.errors = []
        map_config = {"width": "large", "height": 30}
        validator._validate_map_builder(map_config)
        assert len(validator.errors) == 1
        assert "width" in validator.errors[0]
        
        # Very large dimensions (warning)
        validator.errors = []
        validator.warnings = []
        map_config = {"width": 150, "height": 200}
        validator._validate_map_builder(map_config)
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 2
    
    def test_validate_object_counts(self, validator):
        """Test validation of object counts."""
        # Valid counts
        objects = {"wall": 10, "mine": 5}
        validator._validate_object_counts(objects)
        assert len(validator.errors) == 0
        
        # Invalid count type
        validator.errors = []
        objects = {"wall": "many", "mine": 5}
        validator._validate_object_counts(objects)
        assert len(validator.errors) == 1
        assert "non-negative integer" in validator.errors[0]
        
        # Very high total (warning)
        validator.errors = []
        validator.warnings = []
        objects = {"wall": 500, "mine": 600}
        validator._validate_object_counts(objects)
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 1
        assert "Total object count" in validator.warnings[0]
    
    def test_validate_objects(self, validator):
        """Test validation of object configurations."""
        # Valid object config
        objects = {
            "altar": {
                "cooldown": 1,
                "initial_resource_count": 5
            }
        }
        validator._validate_objects(objects)
        assert len(validator.errors) == 0
        
        # Invalid cooldown
        validator.errors = []
        objects = {
            "altar": {
                "cooldown": -1
            }
        }
        validator._validate_objects(objects)
        assert len(validator.errors) == 1
        assert "cooldown" in validator.errors[0]
        
        # Invalid resource count
        validator.errors = []
        objects = {
            "altar": {
                "initial_resource_count": "unlimited"
            }
        }
        validator._validate_objects(objects)
        assert len(validator.errors) == 1
        assert "initial_resource_count" in validator.errors[0]
    
    def test_validate_file_valid(self, validator, valid_config):
        """Test validating a valid configuration file."""
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_file = f.name
        
        try:
            result = validator.validate_file(temp_file)
            assert result is True
            assert len(validator.errors) == 0
        finally:
            os.unlink(temp_file)
    
    def test_validate_file_invalid(self, validator):
        """Test validating an invalid configuration file."""
        invalid_config = {
            "game": {
                "num_agents": -5,
                "map_builder": {
                    "width": "invalid"
                }
            }
            # Missing 'defaults' field
        }
        
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            result = validator.validate_file(temp_file)
            assert result is False
            assert len(validator.errors) > 0
        finally:
            os.unlink(temp_file)
    
    def test_validate_file_not_found(self, validator):
        """Test validating a non-existent file."""
        result = validator.validate_file("/non/existent/file.yaml")
        assert result is False
        assert len(validator.errors) == 1
        assert "Failed to load" in validator.errors[0]


def test_validate_config_file_function():
    """Test the standalone validate_config_file function."""
    valid_config = {
        "defaults": ["_self_"],
        "game": {"num_agents": 5}
    }
    
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        temp_file = f.name
    
    try:
        result = validate_config_file(temp_file)
        assert result is True
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])