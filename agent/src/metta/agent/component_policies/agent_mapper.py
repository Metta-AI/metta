from metta.agent.component_policies.fast import Fast
from metta.agent.component_policies.latent_attn_small import LatentAttnSmall
from metta.agent.component_policies.latent_attn_med import LatentAttnMed
from metta.agent.component_policies.latent_attn_tiny import LatentAttnTiny

agent_classes = {
    "fast": Fast,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}

