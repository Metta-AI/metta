# Layers
Frequently reused layers are in the lib folder for convenient reuse in your policies. Don't hesitate to write new ones but be sure to add an 'initialize' method to your configs to allow others to auto build with your layers. Ideally, all layers take a tensordict as input and read and write from it using unique keys. This allows you to access any tensor in your policy for use in calculating losses. For these tensors, be sure to update the experience spec which tells the experience buffer what to store.
If the layer has additional requirments like initialization to the environment then it should manage those methods on its own, not expecting the policy object to handle these methods for it, allows the policy to be more agnostic to its parts. Further in the spirit of intelligent layers, layers can manage their own memory if necessary or set a spec for the experience buffer.

# Policies
## Automatically Generate Policies From Configs
You can use Policy to auto build a policy from a config that you build. This config should be composed of other configs, each being a layer's config. It initializes a class that does obs shaping by passing it obs information and it initializes all other layers without passing any args, trusting that the config itself has everything needed to spin up the layer. This config of configs should also should meet the following requirements**:
        - Must have an 'obs_shaper' attribute and it should have an instantiate method that takes an obs_meta dict.
        - Must have an 'instantiate' method that uses itself as the config to instantiate the layer.
        - Must be in the order the network is to be executed.
Policy also calls methods like initialize_to_environment on every layer if that layer has such a method as an attribute.

For an example of this path of policy generation, look at vit_lstm.py

## Custom Policy Classes
On the other hand, you can write your own policy that spins up fully, partially, or not at all from a config(s)! You can spec how every layer gets built, the details of the forward method, when to write to the tensordict or not, etc. Be sure to include handling for initialization_to_env, resetting memory, and any other policy-level methods that come up in the future. For an example of this path, look at fast.py.

# Data Structures
Trainer passes a TensorDict to the policy and expects to recieve one, populated with the keys specified in your losses. The tensordict will input environment observations, an environment ID during rollout, rewards, dones, and truncateds.
