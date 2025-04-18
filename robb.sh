#!/bin/bash
# Script to run multiple training jobs in parallel

# First training job - control_0 experiment
echo "Starting control_0 training job..."
python -m tools.train +hardware=pufferbox run=rwalters.control_0 \
  trainer.env=/env/mettagrid/robb_map \
  +trainer.env_overrides.game.max_size=60 &

# Second training job - curriculum_0 experiment
echo "Starting curriculum_0 training job..."
python -m tools.train +hardware=pufferbox run=rwalters.curriculum_0 \
  trainer.env=/env/mettagrid/robb_map &

# Print a message to let the user know jobs have been started
echo "All training jobs have been started in the background."
echo "Use 'jobs' command to see running jobs or 'fg' to bring a job to foreground."