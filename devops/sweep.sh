#/bin/bash -e

# keep running sweep. until it fails N consecutive times
N=10
consecutive_failures=0
while true; do
    ./devops/run.sh sweep "$@"
    if [ $? -ne 0 ]; then
        consecutive_failures=$((consecutive_failures + 1))
        if [ $consecutive_failures -ge $N ]; then
            echo "Sweep failed $N consecutive times, exiting"
            exit 1
        fi
    else
        consecutive_failures=0
    fi
done

