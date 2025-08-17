from metta.agent.pytorch.agalite import AGaLiTe
from metta.agent.pytorch.agalite_hybrid import AgaliteHybrid
from metta.agent.pytorch.agalite_optimized import AGaLiTeOptimized
from metta.agent.pytorch.agalite_turbo import AGaLiTeTurbo
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    "agalite": AGaLiTe,
    "agalite_hybrid": AgaliteHybrid,
    "agalite_optimized": AGaLiTeOptimized,
    "agalite_turbo": AGaLiTeTurbo,
    "example": Example,
    "fast": Fast,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
