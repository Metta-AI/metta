import math

import torch


def position_embeddings(width, height, embedding_dim=128):
    """
    Creates simple 2D position embeddings with x, y coordinates.

    Generates a grid of positions where each point has normalized coordinates
    in the range [-1, 1]. This provides a basic position representation that
    can be used in neural networks to inject spatial information.

    Args:
        width (int): The width of the grid
        height (int): The height of the grid
        embedding_dim (int): Not used in this function, included for API consistency
            with sinusoidal_position_embeddings

    Returns:
        torch.Tensor: A tensor of shape [width, height, 2] containing normalized
            x, y coordinates for each position in the grid
    """
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    pos_x, pos_y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((pos_x, pos_y), dim=-1)


def sinusoidal_position_embeddings(width, height, embedding_dim=128):
    """
    Creates sinusoidal position embeddings for a 2D grid.

    This function implements sinusoidal position embeddings similar to those used
    in the Transformer architecture, but adapted for 2D spatial positions. It creates
    embeddings by applying sine and cosine functions at different frequencies to the
    x and y coordinates, producing a rich representational space that captures both
    position and relative distances.

    The embeddings include both frequency-based features and the raw normalized coordinates
    as the last two dimensions, providing both high-frequency detail and direct positional
    information.

    Args:
        width (int): The width of the grid
        height (int): The height of the grid
        embedding_dim (int): The dimension of the position embeddings, must be even

    Returns:
        torch.Tensor: A tensor of shape [width, height, embedding_dim] containing
            the sinusoidal position embeddings for each position in the grid

    Raises:
        AssertionError: If embedding_dim is not even
    """
    # Generate a grid of positions for x and y coordinates
    x = torch.linspace(-1, 1, width, dtype=torch.float32)
    y = torch.linspace(-1, 1, height, dtype=torch.float32)
    pos_x, pos_y = torch.meshgrid(x, y, indexing="xy")

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
        embeddings[:, :, -2] = pos_x
        embeddings[:, :, -1] = pos_y

    return embeddings
