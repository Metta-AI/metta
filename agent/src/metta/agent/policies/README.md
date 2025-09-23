# Layers

Frequently reused layers are in the lib folder for convenient reuse in your policies. Don't hesitate to write new ones
but be sure to add an 'initialize' method to your configs to allow others to auto build with your layers. Ideally, all
layers take a tensordict as input and read and write from it using unique keys. This allows you to access any tensor in
your policy for use in calculating losses. For these tensors, be sure to update the experience spec which tells the
experience buffer what to store. If the layer has additional requirments like initialization to the environment then it
should manage those methods on its own, not expecting the policy object to handle these methods for it, allows the
policy to be more agnostic to its parts. Further in the spirit of intelligent layers, layers can manage their own memory
if necessary or set a spec for the experience buffer.

# Policies

## Automatically Generate Policies From Configs

You can use Policy to auto build a policy from a config that you build. This config should be composed of other configs,
each being a layer's config. It initializes a class that does obs shaping by passing it obs information and it
initializes all other layers without passing any args, trusting that the config itself has everything needed to spin up
the layer. This config of configs should also should meet the following requirements\*\*: - Must have an 'obs_shaper'
attribute and it should have an instantiate method that takes an obs_meta dict. - Must have an 'instantiate' method that
uses itself as the config to instantiate the layer. - Must be in the order the network is to be executed. Policy also
calls methods like initialize_to_environment on every layer if that layer has such a method as an attribute.

For an example of this path of policy generation, look at vit_lstm.py

## Custom Policy Classes

On the other hand, you can write your own policy that spins up fully, partially, or not at all from a config(s)! You can
spec how every layer gets built, the details of the forward method, when to write to the tensordict or not, etc. Be sure
to include handling for initialization_to_env, resetting memory, and any other policy-level methods that come up in the
future. For an example of this path, look at fast.py.

# Data Structures

Trainer passes a TensorDict to the policy and expects to recieve one, populated with the keys specified in your losses.
The tensordict will input environment observations, an environment ID during rollout, rewards, dones, and truncateds.

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
