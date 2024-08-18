
import torch
import math

def position_embeddings(width, height, embedding_dim=128):
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
    return torch.stack((pos_x, pos_y), dim=-1)

def sinusoidal_position_embeddings(width, height, embedding_dim=128):
    # Generate a grid of positions for x and y coordinates
    x = torch.linspace(-1, 1, width, dtype=torch.float32)
    y = torch.linspace(-1, 1, height, dtype=torch.float32)
    pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')

    # Prepare to generate sinusoidal embeddings
    assert embedding_dim % 2 == 0, "Embedding dimension must be even."

    # Create a series of frequencies exponentially spaced apart
    freqs = torch.exp2(torch.linspace(0, math.log(embedding_dim // 2 - 1), embedding_dim // 2))

    # Apply sinusoidal functions to the positions
    embeddings_x = torch.cat([torch.sin(pos_x[..., None] * freqs), torch.cos(pos_x[..., None] * freqs)], dim=-1)
    embeddings_y = torch.cat([torch.sin(pos_y[..., None] * freqs), torch.cos(pos_y[..., None] * freqs)], dim=-1)

    # Combine x and y embeddings by summing, you could also concatenate or average
    embeddings = embeddings_x + embeddings_y

    # Add float embeddings
    if embedding_dim >= 2:
        embeddings[:,:,-2] = pos_x
        embeddings[:,:,-1] = pos_y

    return embeddings
