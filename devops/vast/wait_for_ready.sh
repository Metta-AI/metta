#!/bin/bash -e

label=$1
if [[ -z "$label" ]]; then
  echo "Please provide a label"
  exit 1
fi

get_info=$(vastai show instances --raw)
actual_status=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .actual_status")

# Wait for the instance to become ready
while [[ "$actual_status" != "running" ]]; do
  echo "Waiting for instance to become ready... ($actual_status)"
  sleep 5  # Wait for 5 seconds before checking again
  get_info=$(vastai show instances --raw)
  actual_status=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .actual_status")
done
