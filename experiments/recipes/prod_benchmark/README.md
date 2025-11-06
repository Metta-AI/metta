# For baseline control:

1. Record the git hash you are currently on
2. Git diff since the last benchmark for any changes in trainer
3. Hyperparameter sweep with default PPO loss on vit_reset architecture on ICL recipe
4. With new hypers in place, run pilot power analysis (variance.py) to check how many runs are required for this env

# Steps to benchmark a new intervention:

1. Hyperparameter sweep with your new architecture; make sure to align dimensionality of old and new hypers and use 5
   random seeds within a sweep
2. Use paired seeds to run the intervention on ICL, number of runs is determined by pilot power analysis
3. Run statistics script (analysis.py) with run metadata
4. Inspect radial plot to understand qualities of your architecture

What all recipes have in common:

- Same seed enabled for paired runs (environment, weight init, curriculum); seed is not hardcoded it is recorded
- Evaluations enabled
- Curriculum enabled
- Run separately for a new architecture with architecture=vit_reset or trxl
- Number of timesteps set to 5B (after peak learning)
- Domain randomization on (ex: num_sinks = rng.choice(cfg.num_sinks))
- Combat is off
- Sweeps infra is set in each recipe
- 4 node, 4 gpu working per recipe
