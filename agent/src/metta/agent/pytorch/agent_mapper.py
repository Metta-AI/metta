# Commented out Fast import due to missing pufferlib dependency
# from metta.agent.pytorch.fast import Recurrent as Fast
from metta.agent.pytorch.latent_attn_med import Recurrent as LatentAttnMed
from metta.agent.pytorch.latent_attn_small import Recurrent as LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import Recurrent as LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    # "fast": Fast,  # Commented out due to missing pufferlib dependency
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
