docker run -ti \
    -v /home/puffer/daveey/train_dir:/workspace/metta/train_dir \
    --gpus all --ulimit nofile=64000 \
    --ulimit nproc=640000 --shm-size=80g  \
    daveey/metta  /bin/bash
