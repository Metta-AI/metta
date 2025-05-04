#! /bin/zsh -e

# Recommended: use a new conda env to avoid package conflicts.
# SkyPilot requires 3.7 <= python <= 3.11.
conda create -y -n sky python=3.10
conda activate sky

# Choose your cloud:

pip install "skypilot[kubernetes]"
pip install "skypilot[aws]"
pip install "skypilot[lambda]"
