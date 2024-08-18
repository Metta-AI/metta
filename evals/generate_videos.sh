#!/bin/bash -e

experiment=$1
# Check if experiment is set
if [ -z "$experiment" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

shift

behavior_cfgs=$(find configs/env/mettagrid/behaviors/ -name "*.yaml" | sed 's|.*/behaviors\(.*\)\.yaml|behaviors\1|')
train_cfgs="a5_25x25 a20_40x40 a20_r4_40x40 a20_b4_40x40"

for envcfg in $behavior_cfgs  $train_cfgs ; do
    echo "Generating video for $envcfg"
    video_name=${envcfg//\//_}
    python -m tools.run \
        cmd=evaluate \
        experiment=$experiment \
        env=mettagrid/$envcfg \
        framework=sample_factory \
        eval.video_path=train_dir/sample_factory/$experiment/$video_name.mp4 \
        eval.gif_path=train_dir/sample_factory/$experiment/$video_name.gif \
        "$@"
done

cd train_dir/sample_factory/$experiment/ && rm -f replay.mp4 && ln -s a20_b4_40x40.mp4 replay.mp4
