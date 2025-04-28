#!/bin/bash

source ./devops/env.sh
./devops/checkout_and_build.sh

echo "Running sandbox"

while true; do
    echo "Running sandbox"
    ./devops/sandbox.sh
    sleep 100
done
