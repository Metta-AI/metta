from metta.agent.pytorch.agalite import AGaLiTe
from metta.agent.pytorch.agalite_hybrid import AgaliteHybrid
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.full_context import FullContext
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    "agalite": AGaLiTe,
    "agalite_hybrid": AgaliteHybrid,
    "example": Example,
    "fast": Fast,
    "full_context": FullContext,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
