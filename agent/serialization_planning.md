The issue is that people train agents at one point in time and want to use those trained agents later, but this doesn't
work because the agent code has changed in ways that break serialization.

There are two main paths to addressing this. One is to "upgrade" the serialized agents so they work with the current
codebase. Two is to build a splint so that the agent code from the time the agents were trained can be run with
up-to-date "tool" code (ie roughly metta minus metta/agent "tool code"). Within the second path there are two main
routes: an intra-process splint where agents run in an isolated namespace in the same python interpreter as the tools,
and a subprocess splint where models run in a spawned subprocess.

For the "upgrade" path, we'd basically develop ad hoc tools to patch issues like field name changes and minor changes to
agent structure and add some supporting information (say a copy of the codebase, or GitHub SHA code) to checkpoints to
make those tools easier to use. This is an incremental path. However, the second path (splint) more or less makes this
unnecessary.

For the "splint" path, I'll focus first on the isolated namespace solution as it is simpler. I'll start with a strawman
proposal to make the technical issues clear. The straw man is to manipulate sys.path inside the python interpreter to
point the tools to use an old version of the metta.agent code:

    +-----------------------+          +----------------------------+
    | Tools (metta minus    |  <---->  | metta.agent (old codebase) |
    | metta.agent)          |          +----------------------------+
    | (current)             |                    |
    |                       |                    v
    |                       |          +----------------------------+
    |                       | -------> | Third-party libraries       |
    |                       |          | (current)                   |
    +-----------------------+          +----------------------------+

This solution is not robust and it's easy to see why:

1.  Prioritized class loading via sys.path is fragile and could introduce subtle bugs that are difficult to reason
    about.
2.  Calls from (current) tools into (old) metta.agent can break because of code changes
3.  Calls from (old) metta.agent back into (current) tools can break because of code changes
4.  Third party libraries are loaded once and shared between old and new versions of code. They may have been upgraded
    etc.

The proposed solution to #1-#3 is:

- Load the old copy of metta.agent into the process in an isolated namespace using a dynamic class loader like
  Huggingface.Transformers. This will eliminate class name / namespace collisions. This addresses #1 but does not
  address #2 / #3
- Formalize a layer in the code that sits between tools and agents and add controls around it so that it doesn't change
  inadvertently. This would include (in my initial analysis) the Policy class, PolicyArchitecture base classes,
  GameRules, and some other code. The constraint is that tools and agents can only interact through this code. We would
  probably want to store it in a separate directory and deploy it internally as a versioned module.
- Along with the formal code layer we probably want a way of enforcing that all calls between tools and agents go
  through it (say a GitHub checkin check?). Alternatively breaking the code into separate folders would work, and
  separate repos would also work but be annoying.

The solution then looks like this:

    +-----------------------+        +---------------------+         +----------------------------------+
    | Tools (metta minus    |  <-->  | compatibility layer |  <----> | metta.agent (isolated namespace) |
    | metta.agent)          |        +---------------------+         +----------------------------------+
    | (current)             |                                                      |
    |                       |                                                      v
    |                       |                                        +----------------------------+
    |                       | -------------------------------------> | Third-party libraries      |
    |                       |                                        | (current)                  |
    +-----------------------+                                        +----------------------------+

For #4, we have to be careful manually. In my initial look at this the main modules that were shared between tools and
metta.agent are torch and pydantic. We'd need to see how they have historically handled upgrades, what has broken, etc.

For #4 the alternate solution of running models in an isolated process is also possible - we would have to pin our
methods for instantiating models and communicating with them. Then agents and tools can use different versions of shared
libraries. It's a more complicated solution at runtime, but it is more robust to model changes. It can be done later.

In order to build this out, I'd suggest doing a proof of concept first (probably a couple full days) and then building a
production version. If I was going to guess the timeline for the production version I'd say a week or two.

The big question I have is whether the codebase is mature enough for this. For this to work well the compatibility layer
has to be nailed down. But it's changed a lot lately. I think the recent refactors are good and can stand the test of
time, but I don't know what's planned.

**Appendix** Some details in no particular order:

_Dynamic loading / namespace isolation_: I was thinking that namespace isolation would be intractable and problematic,
but it turns out that there are libraries that help support this. I looked at both pytorch.hub and
huggingface.transformers. The main relevant difference between the two is that huggingface.transformers supports
namespace isolation. The pytorch.hub library doesn't support namespace isolation (it just prepends to sys.path). That
solution was so unsuitable that I decided to use it as a strawman.

_Checkpoint file format and loading from checkpoints_: The goal is that we could load a model based on code that is in
the metta GitHub or on the local disk, with all metadata, weights, and references to the codebase in the .mpt file. With
a GitHub reference, the .mpt file would look like this:

    checkpoint.mpt
    ├─ manifest.json              # repo, GitHub SHA
    ├─ policy_architecture.yaml   # pydantic-serialized PolicyArchitecture, version-matching the above
    ├─ weights.safetensors        # state_dict

and then we would be able to load using something like the following:

    agent = DyanmicAgentLoader.from_mpt("checkpoint.mpt")

_Rough plan for proof of concept_

1.  Code structure: Analyze the codebase and define the code that must be shared between model code and application
    code. This code needs to be separated from the codebase, packaged, and versioned. Examples would be the Policy
    interface, config base classes, and the GameRules class.
2.  The policy artifact: Define the manifest and add it to the .mpt file
3.  Loading: Build a wrapper around huggingface.transformers which loads the old code from GitHub or another location
    into an isolated namespace and creates a generic Policy that can delegate to a Policy created by the
    PolicyArchitecture model factory. Modify checkpoint manager code to use this loader.
4.  Probably a few more things will come up here
