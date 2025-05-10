
# Making Changes to MettaAgent
Below is a process meant to support the ongoing evolution of the components that make up a `MettaAgent`—from minor improvements to complete rewrites of layer classes- while maintaining backward compatibility for previously trained agents. In practice, this means that older agents must still be able to run inference and training using the exact class implementations they were trained with, even as newer versions of those classes are introduced.

### Versioning Layer Classes
Each layer class (e.g., `ObsNormalizer`, `ActionEmbedding`, etc.) must include a revision tag in its name using the format `ClassNameRevMM-DD-XX`. For example:
`ObsNormalizerRev05-06-01(nn.Module)`. MM-DD represents the date of revision (month and day). XX is a two-digit index for multiple changes on the same day.

### Deprecation and Lifecycle
When a class is superseded, it should be marked as deprecated by adding `DEPRECATED` at the top of its docstring.

Move deprecated classes below a separator comment in the file (e.g., 
# ---------------- DEPRECATED ----------------).

Classes older than two months should be deleted. Before deletion, consider whether any production agents still reference the class.

### Config Files
The contents of configuration files (e.g., `simple.yaml`) may be updated to point to newer layer class names without changing the file names. Older models will still be able to run as long as their classes are in the same location with a fully qualified name.

## Future Work
The above is contingent on continuing to use `torch.save(model, model.pt)` which is a great way to keep everything we need to continue training. 

We can consider saving models using `torch.jit.script()` for archival reasons if we'd like to run inference evaluations for historical benchmarking. This would also require adding @export to some of the MettaAgent class's functions. We won't be able to train these.
