#!/bin/bash -e

model=$1
repo=$2

python -m sample_factory.huggingface.push_to_hub -r metta-ai/$repo -d ./train_dir/sample_factory/$model
