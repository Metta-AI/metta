# Proposed refactor for agent subpackage

Hi all, as discussed with Robb, here’s a PR to propose a refactor to simplify the agent subpackage.
Below, I’ll outline (1) my understanding of the code’s objectives and (2) the proposed architecture.

### 1. Objectives
The metta-agent framework builds neural network policies dynamically from YAML configuration files.
- **Configurable**: The agent's neural net is defined by a YAML file, which lists each layer and its inputs.

- **Standard PyTorch + custom layers:**  We can use standard PyTorch layers (eg Linear, Conv), and  custom layers (eg ActionEmbedding, ObsEncoding, ObsTokenizer).
- **Shape inference**: Input shapes are inferred from the previous node's output
- **TensorDict for Data Flow**: Layers pass data during the forward pass by reading from and writing to a shared `TensorDict`.
- **Extra Features**: Researchers can conveniently configure `weight initialization`, `weight clipping`, `l2_init regularization`, and `weight analysis`.
- **Environment-Adaptive**: Agent uses correct embedding based on environment’s object names.
- **Persistence**: Easy loading/saving of policies from file and wandb, as well as caching.
- **Backwards Compatibility**: We can load policies saved with older versions of the code.

### 2. Proposed Architecture

The core ideas are to (1) centralize orchestration responsibility into the agent, leaving the modules as dumb as possible, and to (2) cleanly separate the graph creation and verification tasks into logically self-contained steps.

```py
class Agent(nn.Module):
    def __init__(self, components, **kwargs):
        ## 1. Resolve dependencies
        # graph: List[Tuple[str,str]], representing topologically sorted edges in a graph (ie later nodes only depend on earlier ones)
        self.graph = self._make_graph(...)

        ## 2. Infer shapes
        # shapes is a Dict[str,shape], ie every node has a computed input shape now
        self.shapes, self.input_shape_param = self._infer_shapes(...)

        ## 3. Instantiate modules
        self.layers = ...

        ## 4. Store metadata (for clipping, l2_init loss, weight initialization, etc)
        for layer in self.layers:
            ...

        ## 5. Initializes weights
        for layer in self.layers:
            ...

    def _make_graph(self, ...):
        # use topological sort to resolve dependencies

    def _infer_shapes(self, ...):
        # infer shapes using rules-per-class, see below.

    def forward(self, x):
        # execute graph

    # centralized extra functions
    def _clip(self):
        for layer in self.layers:
            if layer.clip > 0:
                ...

    def _stability_analysis(self):
        for layer in self.layers:
            if layer.do_stability_analyze:
                ...

    def _get_l2_init_loss(self):
        ...

    # environment adaptivity
    def adapt_to_environment(self):
        ...

    # persistence
    @classmethod
    def load(cls):
        ...

    def save(self):
        ...
```

For the `shape inference` to work, we require all modules to have a `infer_shapes`-method.

```py
@classmethod
def infer_shapes(cls, input_shapes, **nn_params):
    output_shapes = ...
    input_params = ... # Parameter name for layer constructor, eg 'in_features' for nn.Linear or 'in_channels' for nn.Conv2d
    return output_shapes, input_params

# Example for custom module
class Double(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return torch.cat([x*2, x*2], dim=-1) # eg [2, 2] -> [4, 4, 4, 4]

    @classmethod
    def infer_shapes(cls, input_shapes, **nn_params):
        output_shapes = input_shapes[:-1] + [input_shapes[-1]*2]
        input_params = {'in_features': input_shapes[-1]}
        return output_shapes, input_params

# Example for PyTorch module
@patch_classmethod(nn.Linear)
def infer_shapes(cls, input_shapes, **nn_params):
    out_features = nn_params['out_features']
    out_shapes = input_shapes[:-1] + [out_features]
    in_params = {'in_features': input_shapes[-1]}
    return out_shapes, in_params
```

The `saving`/`loading` code (ie `PolicyState`, `PolicyRecord`, `PolicyStore`) would remain largely as-is.
