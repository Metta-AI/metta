#!/bin/bash

# This runs an infinite loop, which keeps the container running and lets
# users ssh into it to run whatever they want
source ./devops/env.sh
<<<<<<< HEAD
./devops/build_mettagrid.sh
=======
uv run --active --directory mettagrid python setup.py build_ext --inplace
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87

echo "Running sandbox"

while true; do
  echo "Running sandbox"
  sleep 100
done
