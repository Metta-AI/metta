from metta.agent.pytorch.agalite import Agalite
from metta.agent.pytorch.agalite_faithful import AGaLiTeFaithful
from metta.agent.pytorch.agalite_pure import AgalitePure
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    "agalite": AGaLiTeFaithful,  # Faithful AGaLiTe implementation with TransformerWrapper
    "agalite_hybrid": Agalite,  # Hybrid AGaLiTe-LSTM for efficiency (old version)
    "agalite_pure": AgalitePure,  # Pure AGaLiTe transformer (experimental)
    "example": Example,
    "fast": Fast,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
    "latent_attn_tiny": LatentAttnTiny,
}
