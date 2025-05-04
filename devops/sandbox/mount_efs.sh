#!/usr/bin/env bash

# TODO - grab this from SSM parameter store
ENDPOINT=$(aws ssm get-parameter --name /shared-efs/url --output text --query "Parameter.Value")

DIR="/Volumes/metta-efs"
REGION="us-east-1"

sudo mkdir -p "$DIR"

if ! mount | grep -q "$DIR"; then
    echo "Mounting EFS..."
    sudo mount -t efs -o tls -o region="$REGION" "$ENDPOINT": "$DIR"
    echo "Mounted EFS at $DIR."
else
    echo "EFS already mounted at $DIR."
    echo "Run 'sudo umount $DIR' to unmount."
fi
