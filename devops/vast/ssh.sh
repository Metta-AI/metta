#!/bin/bash -e

label=$1
if [[ -z "$label" ]]; then
  echo "Please provide a label"
  exit 1
fi

./devops/vast/wait_for_ready.sh $label

get_info=$(vastai show instances --raw)
host=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .ssh_host")
port=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .ssh_port")

# Copy the .netrc file
scp_cmd="scp -P $port $HOME/.netrc root@$host:/root/.netrc"
echo $scp_cmd
$scp_cmd

ssh_cmd="ssh -p $port root@$host"
echo $ssh_cmd
$ssh_cmd
