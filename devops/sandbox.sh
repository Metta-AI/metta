#!/bin/bash

# This runs an infinite loop, which keeps the container running and lets
# users ssh into it to run whatever they want
source ./devops/setup.env

echo "Running sandbox"

while true; do
  echo "Running sandbox"
  sleep 100
done
