#!/usr/bin/env bash

# TODO - update with the actual EFS endpoint
ENDPOINT="fs-0959f49bf925f0023.efs.us-east-1.amazonaws.com"
DIR="/Volumes/metta-efs"
REGION="us-east-1"

sudo mkdir -p "$DIR"

if ! mount | grep -q "$DIR"; then
    echo "Mounting EFS..."
    sudo mount -t efs -o tls -o region="$REGION" "$ENDPOINT": "$DIR"
else
    echo "EFS already mounted at $DIR."
    echo "Run 'sudo umount $DIR' to unmount."
fi
