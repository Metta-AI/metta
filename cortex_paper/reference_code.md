# Broader context

I work an an AI research lab called Softmax. Our monorepo codebase is called `metta`.

I'm helping my colleague Subho write a NeurIPS paper on Cortex. Cortex is a modular framework he made for building recurrent neural network backbones with scalable agent memory systems.

My specific task with regard to helping him is running some basic evaluations that he's going to include in the paper.

# Rough goal

We want to end up with something like this:

* Pick an environment that tests memory
  * E.g. `assembly_lines`, `cvc_random_maps`
  * Can start with a medium difficulty, adjust if needed
  * Run with couple of seeds for each task/curriculum
* Sweep across Cortex architectures
  * General note on hypers
    * Can start with hidden size 128, number of layers 3, adjust if needed.
    * We can have 5 seeds per variant.
    * We don't need to sweep other hypers like lr and can use default for most, unless there is a need to.
  * First try single-cell architectures
    * Associative memory cells: Ag, X, M
    * Hidden-to-hidden connection cells: S
    * State summaries/SSM cells: A
  * Then try column of cells
    * We think AgSA would be a good combo
    * Would want to ablate various aspects of that column, especially the Axon component
  * Finally, check if ^ suffix to Axonify things has any use, and measure performance under strict TBPTT.
  * Also would be test scaling and sweep across varying column width and stack depth.

# How to run things

- Try to keep all of the code you write inside the `cortex_paper/` folder, though feel free to use sub-folders, when running a particular experiment, to keep all scripts, results, etc. grouped there.
- We use SkyPilot, which lets us have (a) long-lived "sandbox" machines and (b) kick off managed jobs on ephemeral machines.
- I have a Macbook with an M4 Pro chip. But I don't train on it because of PyTorch compatibility issues.

- Sometimes I run things  "sandbox" machines.
  - This is great for running quick experiments and sanity checks.
  - It avoids the overhead of starting a new SkyPilot managed job, which needs to create a new EC2 instance, install `metta` on it, etc.
  - Avoid using `./tools/run.py` directly.
  - Instead, use `./devops/run.sh`, which wraps `./tools/run.py` and configures WandB logging, Datadog logging, multi-GPU support, etc.
  - You can always make ``./devops/run.sh`` use only 1 GPU if you need by passing in `NUM_GPUS=1`.
  - Always include `trainer.total_timesteps=...`, else it will run for a default of 10 billion timesteps which is likely too long.
  - Example:

  ```bash
  ./devops/run.sh
    recipes.experiment.abes.cortex.train
    run=test_run
    stack_builder=recipes.experiment.abes.cortex.build_stack
    stack_cfg='{"pattern": "X", "d_hidden": 64, "num_layers": 3}'
    trainer.total_timesteps=5000
  ```

- You can also kick off a managed SkyPilot job to run them on a separate new ephemeral machine.

  - This is useful when running something for a long time, or wanting to run many things in parallel.
  - To do this, use the `./devops/skypilot/launch.py` script.
  - Example:

  ```bash
    ./devops/skypilot/launch.py
      --gpus=4
      --heartbeat-timeout-seconds=3600
      trainer.total_timesteps=100000000
      recipes.experiment.cvc.cvc_random_maps.train
      heart_buckets={heart_buckets}
      resource_buckets={resource_buckets}
      initial_inventory_buckets={initial_inventory_buckets}
      run={run_name}-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}
      num_cogs={num_cogs}
  ```

- Sometimes my colleagues define experiments in `if name == "main":` inside a `.py` file

  - Then they can just run  `python3 -m recipes.experiement.assembly_lines`.
  - Under the hood, this just invokes `./devops/skypilot/launch.py`.
  - You can see an example in `recipes.experiemnt.cvc.cvc_random_maps`.
  - This can be nice for reproducibility and seeing exactly what you kicked off.


Best GPU set up to train on is 8x L4 GPUs, on a SkyPilot managed job, or on my yatharth-sandbox-3 sandbox. Gets 256k sps on cvc_random_maps training AgSA.

# What environments exist

To evaluate Cortex, we want environments that require agents to have memory within an episode.

Memory-less

* `navigation`
* `arena`
* `abes`

Environments we think exercise memory:

* `cvc_random_maps`
* `assembly_lines`

# Current plan



Things I gotta do


- Kick WandB runs in both environments to estimate training time & whether we learn on default knons

- Understand what the different WandB metrics mean
- In context of a colleague's runs on cvc_random_maps
- And assembly_lines, is it different?
- And the baseline we ran that didn't learn anything
- Did our baseline even show any movement? Did it even finish?
- Understand what convergence means and if these runs exhibit it

- Understand the cvc_random_maps knobs & configuration
- Understand the assembly_lines knobs & configuration
- Adjust knobs for environments
- Iterate until we see learning
- Iterate until we see convergence
- Estimate training time

