The problem is that model code and the code that uses models (i'm going to call this application code both change over
time. what people want to do is train models at time T1 and use them at time T2>T1, but the code has migrated over time.

Both the model code and application code can change over time. In this case we always want to run new application code,
but may want to either:

- use _new_ model code (ie the current repo state) against new application code
- use _old_ model code (ie the code the model was trained on) against new application code

## _**new model code against new application code**_

This is the only possibliity with the current codebase. Having moved the codebase over to safetensors, there are still a
few things that can break now. You can understand these by understanding the model deserialization process with
safetensors:

Safetensors Step 1: the code instantiates an instance of the model class Safetensors Step 2: the code configures this
model class to prepare it for the state_dict copy. because the state_dict copy is optimized (it's a zero copy operation
apparently) we need to set up the state_dict tensors for all model components to have the right dimensions Safetensors
Step 3: copy the state_dict, which is a key => Tensor[] map, into the model

To accomplish steps 1 and 2 here, we construct the empty model using a yaml-serialized PolicyArchitecture instance to
instantiate agents. Then for step 3, we load the model_dict from a safetensors-serialized file. The .mpt zip file that
we serialize for checkpoints contains both the yaml-serialized PolicyArchitecture and the state_dict. So serialization
can break if:

1.  the yaml for PolicyArchitecture for a specific model type changes
2.  if the keys in the state_dict for a model changes (ie if someone renames a field in the components dictionary of an
    nn.Model
3.  if a code change causes the tensor dimensions to change for a model

So, one task to improve model serialization robustness in the new model against new application code case would be to
develop a robust way to migrate mismatched files - if there are comprehensible changes from one version to another, say
if someone changed a field name, we could quickly script a migration script to handle this. In order to make this
maximally robust what i'd do:

1.  in the case of a model load failure, create a diagnostic script that gives a good sense of which of the three issues
    above are responsible for the failure
2.  create ad hoc migration tools to do the migration

Another route, not exclusive to the above, is:

1.  store the github SHA code of the code along with the checkpoint during the training process. if we want to train on
    code that isn't in github-main, we should include a diff from github-main
2.  either manually with cursor, examine the code between the current codebase state and the saved codebase state
3.  either manually or with cursor, generate a migration script from this diff

## _**old model code against new application code**_

I was thinking this would be intractable and problematic, but it turns out that there are libraries that help support
this. In fact dynamic code loading is supported by these frameworks via simple code. I looked at both pytorch.hub and
huggingface.transformers. The main relevant difference between the two is that huggingface.transformers supports
namespace isolation (meaning that libraries loaded using huggingface.transformers are loaded into the python interpreter
using namespaces isolated from the current codebase. This means that it can load models from an old version of the
codebase alongside our new code. The pytorch.hub library doesn't support namespace isolation (it just prepends to
sys.path) and so this is a lot more difficult.

The goal is that we could load a model based on code that is in the metta GitHub or on the local disk, with all
metadata, weights, and references to the codebase in the .mpt file:

    checkpoint.mpt
    ├─ manifest.json              # repo, GitHub SHA
    ├─ policy_architecture.yaml   # pydantic-serialized PolicyArchitecture, version-matching the above
    ├─ weights.safetensors        # state_dict

and then we would be able to load using something like the following:

    agent = DyanmicAgentLoader.from_mpt("checkpoint.mpt")

The following lists approximately what we would need to build, how it would work, and the ways it could still break.

In order for this to work we need to build a few things:

1.  Code structure: Analyze the codebase and define the code that must be shared between model code and application
    code. This code needs to be separated from the codebase, packaged, and versioned. Examples would be the Policy
    interface, config base classes, and the GameRules class. Changes to these classes could still break compatibility,
    so you'd want these classes to be stable before proceeding.
2.  The policy artifact: Define the manifest and add it to the .mpt file
3.  Loading: Build a wrapper around huggingface.transformers which loads the old code from GitHub or another location
    into an isolated namespace and creates a generic Policy that can delegate to a Policy created by the
    PolicyArchitecture model factory.
4.  After this we'll have to do some code analysis - if there are references in the tools to specific class names for
    models or PolicyArchitecture instances etc rather than base classes these will need to be analyzed

Caveats:

- We need to get the shared code bit correct as the model code can't reference any metta code outside of this. It would
  be safer to put the model code in a separate directory to discourage such code refs.
- Any shared libraries (torch, pydantic, typing, etc) are not isolated and so have to be compatible with both the old
  version of the model and the new code. If shared libraries change and become incompatible, you could get strange
  behavior when loading old models. (The ABI for sending data between C++ and Python isn't included in this - that's
  handled in the application code.) Specifically with torch, we would want to understand how upgrading could cause
  things to break.

Recommendation:

1. Decide if this makes sense at all, especially isolating and versioning shared code in the metta repo. If you want
   more robustness around safetensors like in the first section, probably do that first? - it's orthogonal work in any
   case.
2. Build out a feature-complete proof of concept / sketch
3. Build out a production version
