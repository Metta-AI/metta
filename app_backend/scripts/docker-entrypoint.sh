#!/bin/bash
# Rewrite kubeconfig to use host.docker.internal instead of 127.0.0.1
if [ -f /root/.kube/config ]; then
    mkdir -p /tmp/.kube
    sed 's/127\.0\.0\.1/host.docker.internal/g' /root/.kube/config > /tmp/.kube/config
    export KUBECONFIG=/tmp/.kube/config
fi

exec "$@"
