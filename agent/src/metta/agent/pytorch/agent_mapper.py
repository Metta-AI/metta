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
    # AGaLiTe implementations (Approximate Gated Linear Transformer)
    "agalite": AGaLiTeFaithful,  # Main AGaLiTe: Faithful port from JAX with full transformer memory management across BPTT sequences
    "agalite_faithful": AGaLiTeFaithful,  # Alias for clarity - exact AGaLiTe architecture with proper recurrent state handling
    "agalite_simple": AGaLiTeSimple,  # Simplified AGaLiTe: No persistent memory between episodes, easier to debug
    "agalite_hybrid": Agalite,  # Original hybrid: Uses LSTM backbone with AGaLiTe-inspired components for 200k steps/sec
    "agalite_pure": AgalitePure,  # Pure transformer AGaLiTe: Experimental version without LSTM, needs infrastructure changes
    
    # Other agent architectures
    "example": Example,  # Simple example agent for testing
    "fast": Fast,  # Fast CNN-based agent optimized for speed
    "latent_attn_tiny": LatentAttnTiny,  # Smallest latent attention model
    "latent_attn_small": LatentAttnSmall,  # Small latent attention model
    "latent_attn_med": LatentAttnMed,  # Medium latent attention model
}
