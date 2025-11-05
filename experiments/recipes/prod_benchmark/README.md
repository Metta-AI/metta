# For baseline control:

1. Record the git hash you are currently on
2. Git diff since the last benchmark for any changes in trainer
3. Hyperparameter sweep with default PPO loss on vit_reset architecture on ICL recipe
4. With new hypers in place, run pilot power analysis (variance.py) to check how many runs are required for this env

# Steps to benchmark a new intervention:

1. Hyperparameter sweep with your new architecture; make sure to align dimensionality of old and new hypers
2. Use paired seeds to run the intervention on ICL, number of runs is determined by pilot power analysis
3. Run statistics script (analysis.py) with run metadata
4. Inspect radial plot to understand qualities of your architecture
