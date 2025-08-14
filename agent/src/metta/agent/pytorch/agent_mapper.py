from metta.agent.pytorch.agalite import Agalite
from metta.agent.pytorch.agalite_faithful import AGaLiTeFaithful
from metta.agent.pytorch.agalite_pure import AgalitePure
from metta.agent.pytorch.agalite_simple import AGaLiTeSimple
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny

# Map from agent type names to their pytorch implementations
agent_classes = {
    # AGaLiTe implementations
    "agalite": AGaLiTeFaithful,  # Faithful port from JAX with proper BPTT handling
    
    # Other experimental AGaLiTe variants (kept for context/testing)
    "agalite_simple": AGaLiTeSimple,  # Simplified version without persistent memory
    "agalite_hybrid": Agalite,  # Original hybrid LSTM version (200k steps/sec)
    "agalite_pure": AgalitePure,  # Pure transformer (experimental)
    
    # Standard agents
    "example": Example,
    "fast": Fast,
    "latent_attn_tiny": LatentAttnTiny,
    "latent_attn_small": LatentAttnSmall,
    "latent_attn_med": LatentAttnMed,
}
