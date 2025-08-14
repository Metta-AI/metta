from metta.agent.pytorch.agalite import AGaLiTe
from metta.agent.pytorch.agalite_hybrid import AgaliteHybrid
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    # AGaLiTe implementations
    "agalite": AGaLiTe,  # Main AGaLiTe implementation with TransformerWrapper for proper BPTT
    "agalite_hybrid": AgaliteHybrid,  # Hybrid AGaLiTe-LSTM version (200k steps/sec)
    # Standard agents
    "example": Example,
    "fast": Fast,
    "latent_attn_tiny": LatentAttnTiny,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
}
