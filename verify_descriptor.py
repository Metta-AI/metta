from mettagrid.policy.policy import PolicySpec, PolicyDescriptor, AgentPolicy
from mettagrid.policy.random_agent import RandomMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.config.mettagrid_config import MettaGridConfig

# 1. Verify PolicySpec handles aliasing
spec = PolicySpec(class_path="my.policy", data_path="weights.pt")
assert spec.policy_class_path == "my.policy"
assert spec.policy_data_path == "weights.pt"
assert spec.descriptor.name == "policy-weights.pt"
print("PolicySpec alias check passed.")

# 2. Verify descriptor plumbing in RandomMultiAgentPolicy
# Create a minimal valid config (empty defaults usually suffice for this test)
cfg = MettaGridConfig()
pei = PolicyEnvInterface.from_mg_cfg(cfg)

# Create policy with explicit descriptor
desc = PolicyDescriptor(name="test-random-policy")
policy = RandomMultiAgentPolicy(pei)
policy.set_descriptor(desc)

# Verify agent policy inherits it
agent_p = policy.agent_policy(0)
assert agent_p.descriptor.name == "test-random-policy"
print("PolicyDescriptor plumbing check passed.")

