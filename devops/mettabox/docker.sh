#!/bin/bash

# Default values
username="mettaai"  # replace with your Docker Hub username
dockerfile=""  # Dockerfile to use
image="metta"
tag="latest"
name="metta"

# Function for building Docker image
build() {
    # Verify a Dockerfile was provided
    if [ -z "$dockerfile" ]; then
        echo "You must specify a Dockerfile with -d."
        exit 1
    fi
    # Check if a Docker container with the same name already exists
    if [ "$(docker ps -aq -f name=^/${name}$)" ]; then
        # Stop and remove the existing container
        echo "A Docker container with the name ${name} already exists. Stopping and removing it..."
        docker stop ${name}
        docker rm ${name}
    fi
    echo "Building Docker image ${username}/${image}:${tag} with Dockerfile ${dockerfile}..."
    docker build -t ${username}/${image}:${tag} -f ${dockerfile} .
}

# Function for testing Docker image
# Need this on ubuntu for x11: xhost +local:docker
test() {
    # Check if a Docker container with the same name already exists
    if [ "$(docker ps -aq -f name=^/${name})" ]; then
        # If the container exists and is stopped, start it
        echo "A Docker container with the name ${name} already exists. Starting it..."
        docker start ${name}
    else
        # If the container does not exist, run a new one
        echo "Running Docker image ${username}/${image}:${tag} and executing shell..."
        docker run -it \
            --name ${name} \
            --network host \
            --gpus all \
            --ulimit nofile=64000 \
            --ulimit nproc=640000 \
            --shm-size=80g \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v /mnt/wslg:/mnt/wslg \
            -v "$(pwd):/puffertank/docker" \
            -v /home/metta/data_dir:/workspace/metta/train_dir \
            --env-file ~/.metta_env \
            -e DISPLAY \
            -e WAYLAND_DISPLAY \
            -e XDG_RUNTIME_DIR \
            -e PULSE_SERVER \
            -e METTA_HOST=$(hostname) \
            -e METTA_USER=$SSH_USER \
            -e WANDB_API_KEY=$WANDB_API_KEY \
            ${username}/${image}:${tag} bash -c "tmux"
    fi
    # Attach to the running container
    docker exec -it -e METTA_HOST=$(hostname) -e METTA_USER=$SSH_USER -e WANDB_API_KEY=$WANDB_API_KEY ${name} bash -c "tmux attach || tmux"
}

# Function for pushing Docker image
push() {
    echo "Pushing Docker image ${username}/${name}:${tag}..."
    docker push ${username}/${name}:${tag}
}

# Function for displaying usage instructions
usage() {
    echo "Usage: $0 command [-d dockerfile] [-n name] [-i image] [-t tag] [-u username]"
    echo "Commands:"
    echo "  build"
    echo "  test"
    echo "  push"
}

# Main script
if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi

command=$1
shift

# Parse command-line arguments for Dockerfile, name, tag, and username
while getopts n:i:t:u:d: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        i) image=${OPTARG};;
        t) tag=${OPTARG};;
        u) username=${OPTARG};;
        d) dockerfile=${OPTARG};;
    esac
done

case $command in
    build)
        build
        ;;
    test)
        test
        ;;
    push)
        push
        ;;
    *)
        usage
        ;;
esac
