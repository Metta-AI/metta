#!/bin/bash

# Usage: ./run_task.sh <start-iterator> <end-iterator>
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start-iterator> <end-iterator>"
    exit 1
fi

start=$1
end=$2

# Check if start and end are valid numbers
if ! [[ $start =~ ^[0-9]+$ && $end =~ ^[0-9]+$ ]]; then
    echo "Error: start and end iterators must be integers."
    exit 1
fi

# Calculate the number of iterations
n=$((end - start + 1))
if [ $n -le 0 ]; then
    echo "Error: end must be greater than or equal to start."
    exit 1
fi

# Run the command n times with the iterator appended to the run name
for ((i = start; i <= end; i++)); do
    python -m devops.aws.launch_task \
    --cmd=train \
    --run=b.alex.matters.0000101.$i \
    --git_branch=alex-epi-init \
    --agent=simple_matters_2 \
    --env.game.actions.attack.cost=251 \
    --env/mettagrid@env=alex_a20_train \
    --agent.actor.epi_init=true

done
