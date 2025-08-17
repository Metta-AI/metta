from metta.agent.pytorch.agalite import AGaLiTe
from metta.agent.pytorch.agalite_hybrid import AgaliteHybrid
from metta.agent.pytorch.agalite_optimized import AGaLiTeOptimized
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.transformer import Transformer
from metta.agent.pytorch.transformer_improved import TransformerImproved
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    "agalite": AGaLiTe,
    "agalite_hybrid": AgaliteHybrid,
    "agalite_optimized": AGaLiTeOptimized,
    "example": Example,
    "fast": Fast,
    "transformer": Transformer,
    "transformer_improved": TransformerImproved,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
