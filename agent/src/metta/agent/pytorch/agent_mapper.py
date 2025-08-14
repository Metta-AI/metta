from metta.agent.pytorch.agalite import Agalite
from metta.agent.pytorch.agalite_simple import AGaLiTeSimple
from metta.agent.pytorch.agalite_pure import AgalitePure
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    "agalite": AGaLiTeSimple,  # Simplified AGaLiTe that works with Metta batching
    "agalite_hybrid": Agalite,  # Hybrid AGaLiTe-LSTM for efficiency (old version)
    "agalite_pure": AgalitePure,  # Pure AGaLiTe transformer (experimental)
    "example": Example,
    "fast": Fast,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
