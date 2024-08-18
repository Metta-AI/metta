#!/bin/bash

label=$1
experiment=$2
get_info=$(vastai show instances --raw)
id=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .id")
cmd="vastai copy $id:/workspace/metta/replays/ ./train_dir/replays/"
echo $cmd
$cmd
cmd="python -m util.render_replay_to_video --replay=./train_dir/replays/0_0.replay --video=./train_dir/replays/0_0.mp4 --fps=5"
echo $cmd
$cmd
