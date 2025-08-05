from metta.agent.pytorch.fast import Recurrent as Fast
from metta.agent.pytorch.latent_attn_med import Recurrent as LatentAttnMed
from metta.agent.pytorch.latent_attn_small import Recurrent as LatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import Recurrent as LatentAttnTiny


agent_classes = {
    "fast.py": Fast,
    "latent_attn_small.py": LatentAttnSmall,
    "latent_attn_med.py": LatentAttnMed,
    "latent_attn_tiny.py": LatentAttnTiny,
}
