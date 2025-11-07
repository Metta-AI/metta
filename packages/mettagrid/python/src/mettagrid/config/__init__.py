from __future__ import annotations

import importlib

__all__ = [
    "ActionConfig",
    "ActionsConfig",
    "AgentConfig",
    "AgentRewards",
    "AssemblerConfig",
    "AttackActionConfig",
    "ChangeVibeActionConfig",
    "ClipperConfig",
    "convert_to_cpp_game_config",
    "GameConfig",
    "GlobalObsConfig",
    "MettaGridConfig",
    "ProtocolConfig",
    "WallConfig",
]

_config_module = importlib.import_module("mettagrid.config.mettagrid_config")
_c_config_module = importlib.import_module("mettagrid.config.mettagrid_c_config")

ActionConfig = _config_module.ActionConfig
ActionsConfig = _config_module.ActionsConfig
AgentConfig = _config_module.AgentConfig
AgentRewards = _config_module.AgentRewards
AssemblerConfig = _config_module.AssemblerConfig
AttackActionConfig = _config_module.AttackActionConfig
ChangeVibeActionConfig = _config_module.ChangeVibeActionConfig
ClipperConfig = _config_module.ClipperConfig
GameConfig = _config_module.GameConfig
GlobalObsConfig = _config_module.GlobalObsConfig
MettaGridConfig = _config_module.MettaGridConfig
ProtocolConfig = _config_module.ProtocolConfig
WallConfig = _config_module.WallConfig

convert_to_cpp_game_config = _c_config_module.convert_to_cpp_game_config
