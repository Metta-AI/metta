#!/bin/bash
# Kind setup for app_backend with Helm deployment

set -e

CLUSTER_NAME="metta-app-backend"
NAMESPACE="default"
BACKEND_RELEASE="metta-app-backend"
ORCHESTRATOR_RELEASE="metta-orchestrator"
BACKEND_CHART="../devops/charts/observatory-backend"
ORCHESTRATOR_CHART="../devops/charts/orchestrator"
BACKEND_VALUES="../devops/charts/observatory-backend/values-local.yaml"
ORCHESTRATOR_VALUES="../devops/charts/orchestrator/values-local.yaml"
APP_BACKEND_IMAGE="metta-app-backend:local"
METTA_LOCAL_IMAGE="metta-local:latest"

# Function to get WANDB API key from .netrc
get_wandb_api_key() {
    if [ -f ~/.netrc ]; then
        awk '/machine api\.wandb\.ai/ {getline; if ($1 == "login") {login=$2; getline; if ($1 == "password") print $2}}' ~/.netrc
    fi
}

# Function to build Docker images
build_docker_images() {
    echo "Building app_backend Docker image..."
    docker build -t $APP_BACKEND_IMAGE -f Dockerfile ..
    
    # Check if metta-local image exists, if not build it
    if ! docker image inspect $METTA_LOCAL_IMAGE >/dev/null 2>&1; then
        echo "Building metta-local image..."
        (cd .. && metta local build-docker-img)
    fi
}

# Function to create or verify Kind cluster
create_or_verify_cluster() {
    if ! kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        echo "Creating Kind cluster..."
        kind create cluster --config kind-cluster.yaml
    else
        # Verify cluster is healthy
        if ! kubectl cluster-info --context kind-${CLUSTER_NAME} >/dev/null 2>&1; then
            echo "Cluster exists but is not healthy. Recreating..."
            kind delete cluster --name ${CLUSTER_NAME}
            kind create cluster --config kind-cluster.yaml
        else
            echo "Using existing Kind cluster: ${CLUSTER_NAME}"
        fi
    fi
    
    # Set kubectl context
    kubectl config use-context kind-${CLUSTER_NAME}
}

case "${1:-help}" in
    "build")
        echo "=== Building Docker images ==="
        build_docker_images
        ;;
        
    "up")
        echo "=== Starting app_backend and orchestrator on Kind ==="
        
        # Create or verify cluster
        create_or_verify_cluster
        
        # Build images if they don't exist
        if ! docker image inspect $APP_BACKEND_IMAGE >/dev/null 2>&1; then
            echo "Building app_backend image..."
            docker build -t $APP_BACKEND_IMAGE -f Dockerfile ..
        fi
        
        if ! docker image inspect $METTA_LOCAL_IMAGE >/dev/null 2>&1; then
            echo "Building metta-local image..."
            (cd .. && metta local build-docker-img)
        fi
        
        # Load images into Kind
        echo "Loading Docker images into Kind..."
        kind load docker-image $APP_BACKEND_IMAGE --name ${CLUSTER_NAME}
        kind load docker-image $METTA_LOCAL_IMAGE --name ${CLUSTER_NAME}
        
        # Get environment variables
        WANDB_API_KEY=${WANDB_API_KEY:-$(get_wandb_api_key)}
        DATABASE_URL=${DATABASE_URL:-"postgresql://postgres:password@host.docker.internal:5432/stats"}
        METTA_API_KEY=${METTA_API_KEY:-"test-api-key"}
        
        # Deploy app_backend first
        echo "Deploying app_backend with Helm..."
        helm upgrade --install $BACKEND_RELEASE $BACKEND_CHART \
            -f $BACKEND_VALUES \
            --namespace $NAMESPACE \
            --create-namespace \
            --set localConfig.wandbApiKey="$WANDB_API_KEY" \
            --set localConfig.databaseUrl="$DATABASE_URL" \
            --set localConfig.mettaApiKey="$METTA_API_KEY" \
            --wait
        
        # Deploy orchestrator
        echo "Deploying orchestrator with Helm..."
        helm upgrade --install $ORCHESTRATOR_RELEASE $ORCHESTRATOR_CHART \
            -f $ORCHESTRATOR_VALUES \
            --namespace $NAMESPACE \
            --set localConfig.wandbApiKey="$WANDB_API_KEY" \
            --wait
        
        echo ""
        echo "=== Deployment complete! ==="
        echo "App backend is available at: http://localhost:8080"
        echo ""
        echo "To check status:"
        echo "  kubectl get pods -n $NAMESPACE"
        echo "  kubectl logs -n $NAMESPACE -l app=$BACKEND_RELEASE -f    # Backend logs"
        echo "  kubectl logs -n $NAMESPACE -l app=$ORCHESTRATOR_RELEASE -f # Orchestrator logs"
        echo ""
        echo "To stop: $0 down"
        ;;
        
    "down")
        echo "=== Stopping app_backend and orchestrator ==="
        kubectl config use-context kind-${CLUSTER_NAME}
        
        # Uninstall Helm releases
        helm uninstall $BACKEND_RELEASE --namespace $NAMESPACE || true
        helm uninstall $ORCHESTRATOR_RELEASE --namespace $NAMESPACE || true
        
        # Also delete any eval-worker pods created by orchestrator
        kubectl delete pods -l app=eval-worker --namespace $NAMESPACE || true
        
        echo "Services stopped (cluster preserved for faster restarts)"
        echo "To destroy cluster: $0 clean"
        ;;
        
    "clean")
        echo "=== Cleaning up ==="
        
        # Delete cluster
        kind delete cluster --name ${CLUSTER_NAME} || true
        
        echo "Cluster deleted"
        ;;
        
    "logs")
        SERVICE=${2:-backend}
        echo "=== Showing logs for $SERVICE ==="
        kubectl config use-context kind-${CLUSTER_NAME}
        
        case "$SERVICE" in
            "backend")
                kubectl logs -n $NAMESPACE -l app=$BACKEND_RELEASE -f
                ;;
            "orchestrator")
                kubectl logs -n $NAMESPACE -l app=$ORCHESTRATOR_RELEASE -f
                ;;
            "workers")
                kubectl logs -n $NAMESPACE -l app=eval-worker -f
                ;;
            *)
                echo "Unknown service: $SERVICE"
                echo "Usage: $0 logs {backend|orchestrator|workers}"
                exit 1
                ;;
        esac
        ;;
        
    "status")
        echo "=== Cluster status ==="
        kubectl config use-context kind-${CLUSTER_NAME}
        echo ""
        echo "Backend pods:"
        kubectl get pods -n $NAMESPACE -l app=$BACKEND_RELEASE
        echo ""
        echo "Orchestrator pods:"
        kubectl get pods -n $NAMESPACE -l app=$ORCHESTRATOR_RELEASE
        echo ""
        echo "Worker pods:"
        kubectl get pods -n $NAMESPACE -l app=eval-worker
        echo ""
        echo "Services:"
        kubectl get svc -n $NAMESPACE
        echo ""
        echo "Helm releases:"
        helm list -n $NAMESPACE
        ;;
        
    *)
        echo "Usage: $0 {build|up|down|clean|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker images (app_backend and metta-local)"
        echo "  up      - Start app_backend and orchestrator on Kind"
        echo "  down    - Stop services (preserves cluster)"
        echo "  clean   - Delete Kind cluster"
        echo "  logs    - Show logs (usage: logs {backend|orchestrator|workers})"
        echo "  status  - Show deployment status"
        echo ""
        echo "Environment variables:"
        echo "  WANDB_API_KEY   - WandB API key (auto-detected from ~/.netrc)"
        echo "  DATABASE_URL    - PostgreSQL connection string"
        echo "  METTA_API_KEY   - Metta API key for authentication"
        echo ""
        echo "Services:"
        echo "  - App Backend: http://localhost:8080"
        echo "  - Orchestrator: Manages eval tasks and worker pods"
        exit 1
        ;;
esac