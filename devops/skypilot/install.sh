#! /bin/zsh -e

# Recommended: use a new conda env to avoid package conflicts.
# SkyPilot requires 3.7 <= python <= 3.11.
conda create -y -n sky python=3.10
conda activate sky

# Choose your cloud:

pip install "skypilot[kubernetes]"
pip install "skypilot[aws]"
pip install "skypilot[gcp]"
pip install "skypilot[azure]"
pip install "skypilot[oci]"
pip install "skypilot[lambda]"
pip install "skypilot[runpod]"
pip install "skypilot[fluidstack]"
pip install "skypilot[paperspace]"
pip install "skypilot[cudo]"
pip install "skypilot[ibm]"
pip install "skypilot[scp]"
pip install "skypilot[vsphere]"
pip install "skypilot[nebius]"
pip install "skypilot[all]"
