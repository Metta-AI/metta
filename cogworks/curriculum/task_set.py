from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.rl.env_config import EnvConfig
from .config import (
    TaskSetConfig, 
    WeightedTaskSetConfig, 
    BuckettedTaskSetConfig, 
    WeightedTaskSetItem,
    BucketValue
)

logger = logging.getLogger(__name__)


class TaskSet(ABC):
    """Base class for generating tasks with deterministic seeding.
    
    TaskSet supports .get_task(task_id) where task_id is used as the seed.
    It should always be constructed with a TaskSetConfig.
    """
    
    def __init__(self, config: TaskSetConfig):
        self.config = config
        
    @abstractmethod
    def get_task(self, task_id: int) -> EnvConfig:
        """Generate a task (EnvConfig) using task_id as seed."""
        pass
        
    def _init_rng(self, task_id: int) -> random.Random:
        """Initialize and return a seeded random number generator."""
        rng = random.Random()
        rng.seed(task_id)
        return rng


class WeightedTaskSet(TaskSet):
    """TaskSet that contains a list of EnvConfigs or TaskSets with weights.
    
    When get_task() is called, rng is initialized with seed, then we sample 
    from the list by weight. If it's an EnvConfig, return it, otherwise 
    return child.get_task().
    """
    
    def __init__(self, config: WeightedTaskSetConfig):
        super().__init__(config)
        self.config: WeightedTaskSetConfig = config
        self.items = self._build_items()
        self.overrides = self._parse_overrides(config.overrides)
        
    def _build_items(self) -> list[tuple[EnvConfig | TaskSet, float]]:
        """Build items list from config."""
        items = []
        for item_config in self.config.items:
            if item_config.env_config is not None:
                items.append((item_config.env_config, item_config.weight))
            elif item_config.task_set_config is not None:
                # Create nested TaskSet
                nested_task_set = create_task_set_from_config(item_config.task_set_config)
                items.append((nested_task_set, item_config.weight))
        return items
        
    def _parse_overrides(self, overrides: dict[str, Any] | list[str] | None) -> dict[str, Any]:
        """Parse overrides from nested dict or list of "a.b.c.d: val" key values."""
        if overrides is None:
            return {}
            
        if isinstance(overrides, dict):
            return overrides
            
        if isinstance(overrides, list):
            parsed = {}
            for item in overrides:
                if ":" not in item:
                    logger.warning(f"Invalid override format: {item}, expected 'key: value'")
                    continue
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Try to parse value as number or boolean
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif "." in value:
                        value = float(value)
                    elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                parsed[key] = value
            return parsed
            
        return {}
        
    def _apply_overrides(self, env_config: EnvConfig) -> EnvConfig:
        """Apply overrides to an EnvConfig."""
        if not self.overrides:
            return env_config
            
        # Convert to dict, apply overrides, then back to EnvConfig
        config_dict = env_config.model_dump()
        
        for key, value in self.overrides.items():
            # Handle nested keys like "a.b.c"
            keys = key.split(".")
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            
        return EnvConfig.model_validate(config_dict)
        
    def get_task(self, task_id: int) -> EnvConfig:
        """Sample from items by weight and return EnvConfig."""
        rng = self._init_rng(task_id)
        
        if not self.items:
            raise ValueError("No items to sample from")
            
        # Extract items and weights
        items, weights = zip(*self.items)
        
        # Sample by weight  
        selected_item = rng.choices(items, weights=weights)[0]
        
        # If it's an EnvConfig, apply overrides and return
        if isinstance(selected_item, EnvConfig):
            return self._apply_overrides(selected_item)
            
        # If it's a TaskSet, recursively get task with modified task_id
        if isinstance(selected_item, TaskSet):
            # Use a different seed for nested TaskSets to avoid conflicts
            nested_task_id = task_id + 1000
            task = selected_item.get_task(nested_task_id)
            return self._apply_overrides(task)
            
        raise ValueError(f"Invalid item type: {type(selected_item)}")


class BuckettedTaskSet(TaskSet):
    """TaskSet with buckets that can be values or ranges.
    
    Contains a list of keys and buckets. When returning the task, we use the rng
    to sample the bucket, and if it's a range to also sample from the range.
    """
    
    def __init__(self, config: BuckettedTaskSetConfig):
        super().__init__(config)
        self.config: BuckettedTaskSetConfig = config
        self.base_config = config.base_config
        self.buckets = self._build_buckets()
        
    def _build_buckets(self) -> dict[str, list[Any | tuple[Any, Any]]]:
        """Build buckets from config."""
        buckets = {}
        for key, bucket_values in self.config.buckets.items():
            bucket_list = []
            for bucket_value in bucket_values:
                if bucket_value.value is not None:
                    bucket_list.append(bucket_value.value)
                elif bucket_value.range_min is not None and bucket_value.range_max is not None:
                    bucket_list.append((bucket_value.range_min, bucket_value.range_max))
            buckets[key] = bucket_list
        return buckets
        
    def get_task(self, task_id: int) -> EnvConfig:
        """Generate task by sampling from buckets."""
        rng = self._init_rng(task_id)
        
        # Start with base config
        config_dict = self.base_config.model_dump()
        
        # Sample from each bucket
        for key, bucket_values in self.buckets.items():
            if not bucket_values:
                continue
                
            # Sample a bucket value
            bucket_value = rng.choice(bucket_values)
            
            # If it's a tuple (range), sample from the range
            if isinstance(bucket_value, tuple) and len(bucket_value) == 2:
                min_val, max_val = bucket_value
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        sampled_value = rng.randint(min_val, max_val)
                    else:
                        sampled_value = rng.uniform(min_val, max_val)
                else:
                    sampled_value = bucket_value  # Use tuple as-is if not numeric
            else:
                sampled_value = bucket_value
                
            # Apply to config using nested key syntax
            keys = key.split(".")
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = sampled_value
            
        return EnvConfig.model_validate(config_dict)


def create_task_set_from_config(config: TaskSetConfig | DictConfig) -> TaskSet:
    """Create a TaskSet from configuration."""
    if isinstance(config, TaskSetConfig):
        # Handle Pydantic config
        if isinstance(config, WeightedTaskSetConfig):
            return WeightedTaskSet(config)
        elif isinstance(config, BuckettedTaskSetConfig):
            return BuckettedTaskSet(config)
        elif type(config) == TaskSetConfig:
            # Base TaskSetConfig, try to convert to WeightedTaskSetConfig with empty items
            weighted_config = WeightedTaskSetConfig(items=[])
            return WeightedTaskSet(weighted_config)
        else:
            raise ValueError(f"Unknown TaskSetConfig type: {type(config)}")
    
    else:
        # Handle legacy Hydra configuration
        task_set_type = config.get("_target_", "WeightedTaskSet")
        
        if task_set_type == "WeightedTaskSet" or "_target_" not in config:
            items = []
            for item_config in config.get("items", []):
                if "env_config" in item_config:
                    env_config = EnvConfig.model_validate(item_config["env_config"])
                    weight = item_config.get("weight", 1.0)
                    items.append(WeightedTaskSetItem(env_config=env_config, weight=weight))
                elif "task_set" in item_config:
                    child_config = create_task_set_config_from_dict(item_config["task_set"])
                    weight = item_config.get("weight", 1.0)
                    items.append(WeightedTaskSetItem(task_set_config=child_config, weight=weight))
                    
            overrides = config.get("overrides")
            weighted_config = WeightedTaskSetConfig(items=items, overrides=overrides)
            return WeightedTaskSet(weighted_config)
            
        elif task_set_type == "BuckettedTaskSet":
            base_config = EnvConfig.model_validate(config["base_config"])
            buckets_dict = OmegaConf.to_container(config["buckets"])
            
            # Convert buckets to BucketValue format
            buckets = {}
            for key, values in buckets_dict.items():
                bucket_values = []
                for value in values:
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        bucket_values.append(BucketValue(range_min=value[0], range_max=value[1]))
                    else:
                        bucket_values.append(BucketValue(value=value))
                buckets[key] = bucket_values
                
            bucketed_config = BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
            return BuckettedTaskSet(bucketed_config)
            
        else:
            raise ValueError(f"Unknown TaskSet type: {task_set_type}")


def create_task_set_config_from_dict(config_dict: dict) -> TaskSetConfig:
    """Create a TaskSetConfig from a dictionary (helper for legacy support)."""
    task_set_type = config_dict.get("_target_", "WeightedTaskSet")
    
    if task_set_type == "WeightedTaskSet" or "_target_" not in config_dict:
        items = []
        for item_config in config_dict.get("items", []):
            if "env_config" in item_config:
                env_config = EnvConfig.model_validate(item_config["env_config"])
                weight = item_config.get("weight", 1.0)
                items.append(WeightedTaskSetItem(env_config=env_config, weight=weight))
            elif "task_set" in item_config:
                child_config = create_task_set_config_from_dict(item_config["task_set"])
                weight = item_config.get("weight", 1.0)
                items.append(WeightedTaskSetItem(task_set_config=child_config, weight=weight))
                
        overrides = config_dict.get("overrides")
        return WeightedTaskSetConfig(items=items, overrides=overrides)
        
    elif task_set_type == "BuckettedTaskSet":
        base_config = EnvConfig.model_validate(config_dict["base_config"])
        buckets_dict = config_dict["buckets"]
        
        # Convert buckets to BucketValue format
        buckets = {}
        for key, values in buckets_dict.items():
            bucket_values = []
            for value in values:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    bucket_values.append(BucketValue(range_min=value[0], range_max=value[1]))
                else:
                    bucket_values.append(BucketValue(value=value))
            buckets[key] = bucket_values
            
        return BuckettedTaskSetConfig(base_config=base_config, buckets=buckets)
        
    else:
        raise ValueError(f"Unknown TaskSet type: {task_set_type}")