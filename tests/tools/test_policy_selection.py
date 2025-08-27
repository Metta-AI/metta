# import pytest

# from metta.mettagrid import dtype_observations
# from metta.mettagrid.mettagrid_config import MettaGridConfig
# from metta.mettagrid.mettagrid_env import MettaGridEnv
# # from tools.renderer import OpportunisticPolicy, RandomPolicy, RendererToolConfig, SimplePolicy, get_policy

# TODO: (richard) #dehydration
# @pytest.fixture
# def tiny_env():
#     env_cfg = MettaGridConfig.EmptyRoom(num_agents=1)
#     env = MettaGridEnv(env_cfg, render_mode="human")
#     obs, _ = env.reset()
#     assert obs.dtype == dtype_observations
#     try:
#         yield env
#     finally:
#         env.close()


# def test_get_policy_selection_basic(tiny_env):
#     rt_config = RendererToolConfig(policy_type="random")
#     assert isinstance(get_policy("random", tiny_env, rt_config), RandomPolicy)
#     assert isinstance(get_policy("simple", tiny_env, rt_config), SimplePolicy)
#     assert isinstance(get_policy("opportunistic", tiny_env, rt_config), OpportunisticPolicy)
#     # Unknown falls back to simple
#     assert isinstance(get_policy("does_not_exist", tiny_env, rt_config), SimplePolicy)
