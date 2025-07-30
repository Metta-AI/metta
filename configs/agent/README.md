# Architecture Overview 06-18-2025

## CNN-based Encoder Models

fast.yaml uses a token-to-box converter that takes environment-generated tokens, converts them into a tensor that
represents the tokens in a box with channels, width, and height, and then passes it to a CNN encoder stack. It's super
fast and expressive but not quite as expressive as the attention-based architectures and is not robust when the features
in the environment change - you'll have to train a new model.

## Attention-based Encoder Models

latent_attn_med, latent_attn_small, and latent_attn_tiny are all fully robust agents, flexible to changing observation
features and changing action spaces. All use the same action policy network. Their memory units are also all the same.
Their observation encoders also use the same initial stack: stripping some padding tokens from the environment tokens
and using a Fourier feature representation for position encoding, which perform better than learnable encodings.
Finally, they pass the encoded sequence of environment tokens to attention mechanisms, all of which are multi-headed.
Their differences are in their size and complexity which affect their speed and sample-efficiency.

### latent_attn_tiny.yaml

Uses a single learnable token as a query against the entire sequence of encoded observations. It's the fastest to run
and is robust to changing environments but is not as expressive as any of the other models including fast.yaml.

### latent_attn_small.yaml

This is the sweet spot of performance and expressivity. It uses latent attention: a small number of learnable tokens
attend to the entire sequence of encoded obs tokens in a cross attention layer. Those latent outputs then all attend to
each other in a self-attention layer. We inject a classifier token and then output it as the summary output to the LSTM.
It runs a bit slower than latent_attn_tiny.yaml.

### latent_attn_med.yaml

Same as above but with more layers. It seems to be the most sample efficient model but suffers from the worst wall-clock
performance for obvious reasons. It runs at roughly half the speed of latent_attn_tiny.yaml.

# Hard-earned Lessons

- Embedding representations with more than roughly 48 dimensions struggle to learn at our sparsity. Try to keep dims
  around that or below before moving into higher layers. For instance, run a couple of layers at 32 then try to move to
  another block with higher representation dim. This may not be true if kickstarting is used.
- Positional encoding is high-yield. We don't have enough samples to quickly infer position from learnable embeddings.
  In experiments, the prior injected by Fourier features significantly improved performance over both learnable
  embeddings and RoPE. Further, moving from four frequencies up to eight also helped although this is confounded by the
  expanded size of the observation embedding.

This architecture search is far from complete; there are a number of additional techniques to try!
