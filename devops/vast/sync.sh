#!/bin/bash

label=$1
experiment=$2
get_info=$(vastai show instances --raw)
id=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .id")
cmd="vastai copy $id:/workspace/metta/train_dir/$experiment ./train_dir/"
echo $cmd
$cmd
sudo chmod -R 755 ./train_dir/$experiment
