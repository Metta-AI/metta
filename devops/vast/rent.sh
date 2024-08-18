#!/bin/bash -e

label=$1
machine=$(
    ./devops/vast/search.sh \
    | tail -n1 \
    | awk '{print $1}')

echo "Reserving $machine with label $label"

cmd="vastai create instance $machine \
   --image daveey/metta:latest \
   --disk 60 \
   --onstart-cmd '/bin/bash' \
   --label $label \
   --ssh --direct \
   --args --ulimit nofile=unlimited --ulimit nproc=unlimited -c 'echo hello; sleep infinity;' \
   "


echo $cmd
$cmd
