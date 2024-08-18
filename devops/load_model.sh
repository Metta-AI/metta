#!/bin/bash -e

model=$1

if [ -z "$model" ]; then
    model="baseline"
fi

python -m sample_factory.huggingface.load_from_hub -r metta-ai/$model -d ./train_dir/sample_factory

cd train_dir/sample_factory/$model
cp_path=$(ls checkpoint_p0/*.pth | tail -n 1)
echo "Linking $cp_path to train_dir/sample_factory/$model/latest.pth"
ln -s $cp_path latest.pth
