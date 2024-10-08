#/bin/bash -e

# Extract run_id from arguments
run_id=""
for arg in "$@"; do
    if [[ $arg == run=* ]]; then
        run_id="${arg#run=}"
        brea
    fi
done



if [ -z "$run_id" ]; then
    echo "Error: run_id not provided. Please include run=<run_id> in the arguments."
    exit 1
fi

echo "Evolving run: $run_id"

./devops/run.sh sweep train.init_policy_uri="wandb://sweep_model/$run_id@top.trained_policy_elo" "$@"
