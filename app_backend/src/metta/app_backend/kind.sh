#!/bin/bash
# Simple Kind setup for Kubernetes testing - just runs orchestrator

set -e

CLUSTER_NAME="metta-local"

get_wandb_api_key() {
    if [ -f ~/.netrc ]; then
        awk '/machine api\.wandb\.ai/ {getline; if ($1 == "login") {login=$2; getline; if ($1 == "password") print $2}}' ~/.netrc
    fi
}


case "${1:-up}" in
    "build")
        # Create cluster if needed
        if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
            echo "Creating Kind cluster..."
            kind create cluster --name ${CLUSTER_NAME}
        else
            # Verify cluster is healthy
            if ! kubectl cluster-info --context kind-${CLUSTER_NAME} >/dev/null 2>&1; then
                echo "Cluster exists but is not healthy. Recreating..."
                kind delete cluster --name ${CLUSTER_NAME}
                kind create cluster --name ${CLUSTER_NAME}
            fi
        fi

        # Set kubectl context
        kubectl config use-context kind-${CLUSTER_NAME}

        # Load metta-local image
        if ! docker image inspect metta-local:latest >/dev/null 2>&1; then
            echo "Building metta-local image..."
            metta local build-docker-img
        fi
        echo "Loading metta-local:latest into Kind..."
        kind load docker-image metta-local:latest --name ${CLUSTER_NAME}

        # Create RBAC for pod management
        kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: default
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-manager
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "create", "delete", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: default-pod-manager
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: pod-manager
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
EOF
        ;;
    "up")
        kubectl config use-context kind-${CLUSTER_NAME}
        if [ -z "$WANDB_API_KEY" ]; then
            WANDB_API_KEY=$(get_wandb_api_key)
        fi

        # Run orchestrator pod
        kubectl run orchestrator \
            --image=metta-local:latest \
            --image-pull-policy=Never \
            --env="CONTAINER_RUNTIME=k8s" \
            --env="KUBERNETES_NAMESPACE=default" \
            --env="DOCKER_IMAGE=metta-local:latest" \
            --env="WANDB_API_KEY=${WANDB_API_KEY}" \
            --env="BACKEND_URL=${BACKEND_URL:-http://host.docker.internal:8000}" \
            --restart=Never \
            --command -- uv run python -m metta.app_backend.eval_task_orchestrator

        echo "Orchestrator running in Kind"
        echo ""
        echo "Using backend at: ${BACKEND_URL:-http://host.docker.internal:8000}"
        echo ""
        echo "To view orchestrator logs: kubectl logs orchestrator -f"
        echo "To view pods: kubectl get pods -w"
        echo "To stop: ./kind.sh down"
        ;;

    "down")
        echo "Stopping..."
        kubectl config use-context kind-${CLUSTER_NAME}
        kubectl delete pod orchestrator --ignore-not-found=true
        kubectl delete pods -l app=eval-worker --ignore-not-found=true
        echo "Stopped (cluster preserved for faster restarts)"
        ;;

    "clean")
        echo "Deleting cluster..."
        kubectl config use-context kind-${CLUSTER_NAME}
        kind delete cluster --name ${CLUSTER_NAME}
        echo "Cluster deleted"
        ;;
    *)
        echo "Usage: ./kind.sh {build|up|down|clean}"
        exit 1
        ;;
esac
