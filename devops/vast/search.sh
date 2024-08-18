#!/bin/bash -e

vastai search offers num_gpus=1 \
"cpu_cores_effective>8" \
"inet_down>100" \
"inet_up>100" \
"cuda_vers>=12.1" \
gpu_name=RTX_4090 rented=False "dph<1" -o dph-
