#!/usr/bin/env python3
"""
Demonstration of implementing an Asana task from External Projects.
This shows how a typical task might be implemented.
"""

import os
import sys
from datetime import datetime


# Example Asana task that might be fetched from External Projects
EXAMPLE_TASK = {
    "gid": "1234567890",
    "name": "Add configuration validation for mettagrid environments",
    "notes": """
Implement a configuration validator for mettagrid environment YAML files.

Requirements:
1. Validate that all required fields are present
2. Check that object counts are within reasonable limits
3. Ensure map dimensions are valid
4. Validate that referenced objects exist in the objects configuration
5. Add helpful error messages for common configuration mistakes

The validator should be run as part of the test suite and also be available as a standalone tool.
""",
    "assignee": {"name": "External Contributor"},
    "due_on": "2024-02-01",
    "tags": [{"name": "enhancement"}, {"name": "good-first-issue"}],
    "custom_fields": [
        {"name": "Priority", "text_value": "Medium"},
        {"name": "Effort", "number_value": 3}
    ]
}


class MettagridConfigValidator:
    """Validator for mettagrid environment configuration files."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_file(self, config_path: str) -> bool:
        """Validate a mettagrid configuration file."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load config file: {e}")
            return False
        
        # Validate required fields
        self._validate_required_fields(config)
        
        # Validate game configuration
        if 'game' in config:
            self._validate_game_config(config['game'])
        
        # Print results
        self._print_results(config_path)
        
        return len(self.errors) == 0
    
    def _validate_required_fields(self, config: dict):
        """Check that all required fields are present."""
        required_fields = ['defaults', 'game']
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
    
    def _validate_game_config(self, game_config: dict):
        """Validate the game configuration section."""
        # Validate num_agents
        if 'num_agents' in game_config:
            num_agents = game_config['num_agents']
            if not isinstance(num_agents, int) or num_agents <= 0:
                self.errors.append(f"num_agents must be a positive integer, got: {num_agents}")
            elif num_agents > 100:
                self.warnings.append(f"num_agents is very high ({num_agents}), this may cause performance issues")
        
        # Validate map_builder
        if 'map_builder' in game_config:
            self._validate_map_builder(game_config['map_builder'])
        
        # Validate objects
        if 'objects' in game_config:
            self._validate_objects(game_config['objects'])
    
    def _validate_map_builder(self, map_config: dict):
        """Validate map builder configuration."""
        # Validate dimensions
        for dim in ['width', 'height']:
            if dim in map_config:
                value = map_config[dim]
                if not isinstance(value, int) or value <= 0:
                    self.errors.append(f"Map {dim} must be a positive integer, got: {value}")
                elif value > 100:
                    self.warnings.append(f"Map {dim} is very large ({value}), this may cause performance issues")
        
        # Validate instances
        if 'instances' in map_config:
            instances = map_config['instances']
            if not isinstance(instances, int) or instances <= 0:
                self.errors.append(f"instances must be a positive integer, got: {instances}")
        
        # Validate root scene
        if 'root' in map_config:
            root = map_config['root']
            if 'type' not in root:
                self.errors.append("Map root must have a 'type' field")
            
            if 'params' in root and 'objects' in root['params']:
                self._validate_object_counts(root['params']['objects'])
    
    def _validate_object_counts(self, objects: dict):
        """Validate object counts in map configuration."""
        total_objects = 0
        
        for obj_type, count in objects.items():
            if not isinstance(count, int) or count < 0:
                self.errors.append(f"Object count for '{obj_type}' must be a non-negative integer, got: {count}")
            else:
                total_objects += count
        
        if total_objects > 1000:
            self.warnings.append(f"Total object count is very high ({total_objects}), this may cause performance issues")
    
    def _validate_objects(self, objects: dict):
        """Validate object configurations."""
        for obj_name, obj_config in objects.items():
            if isinstance(obj_config, dict):
                # Validate cooldown
                if 'cooldown' in obj_config:
                    cooldown = obj_config['cooldown']
                    if not isinstance(cooldown, (int, float)) or cooldown < 0:
                        self.errors.append(f"Object '{obj_name}' cooldown must be non-negative, got: {cooldown}")
                
                # Validate resource counts
                if 'initial_resource_count' in obj_config:
                    count = obj_config['initial_resource_count']
                    if not isinstance(count, int) or count < 0:
                        self.errors.append(f"Object '{obj_name}' initial_resource_count must be non-negative integer, got: {count}")
    
    def _print_results(self, config_path: str):
        """Print validation results."""
        print(f"\nValidation results for: {config_path}")
        print("-" * 50)
        
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ Configuration is valid!")


def validate_config_file(config_path: str) -> bool:
    """Validate a single configuration file."""
    validator = MettagridConfigValidator()
    return validator.validate_file(config_path)


def validate_all_configs():
    """Validate all mettagrid configuration files."""
    import glob
    
    config_dir = "configs/env/mettagrid"
    config_files = glob.glob(os.path.join(config_dir, "**/*.yaml"), recursive=True)
    
    print(f"Found {len(config_files)} configuration files to validate\n")
    
    all_valid = True
    for config_file in config_files:
        validator = MettagridConfigValidator()
        if not validator.validate_file(config_file):
            all_valid = False
    
    return all_valid


def main():
    """Main entry point for the validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate mettagrid configuration files")
    parser.add_argument("config_file", nargs="?", help="Path to configuration file to validate")
    parser.add_argument("--all", action="store_true", help="Validate all configuration files")
    
    args = parser.parse_args()
    
    if args.all:
        success = validate_all_configs()
    elif args.config_file:
        success = validate_config_file(args.config_file)
    else:
        # Demo mode - show the example task and validate the currently open file
        print("=" * 70)
        print("ASANA TASK IMPLEMENTATION DEMO")
        print("=" * 70)
        print(f"\nTask: {EXAMPLE_TASK['name']}")
        print(f"GID: {EXAMPLE_TASK['gid']}")
        print(f"\nDescription:\n{EXAMPLE_TASK['notes']}")
        print("\n" + "=" * 70)
        print("\nImplementation: Configuration Validator for Mettagrid Environments")
        print("=" * 70)
        
        # Validate the file that's currently open
        test_file = "configs/env/mettagrid/arena/tag.yaml"
        if os.path.exists(test_file):
            print(f"\nValidating example file: {test_file}")
            success = validate_config_file(test_file)
        else:
            print("\nNo configuration file found to validate")
            success = True
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())