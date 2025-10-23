#!/usr/bin/env bash
# Deploy a Helm chart to Kubernetes with customizable options

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHART_PATH=""
RELEASE_NAME=""
NAMESPACE=""
CLUSTER_NAME="main"
REGION="us-east-1"
DRY_RUN=false
WAIT=true
TIMEOUT="5m"
CREATE_NAMESPACE=false
EXTRA_ARGS=()

# Usage information
usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Deploy a Helm chart to Kubernetes/EKS cluster.

Required Options:
  --chart PATH          Path to Helm chart directory
  --release NAME        Helm release name
  --namespace NS        Kubernetes namespace

Optional Options:
  --cluster NAME        EKS cluster name (default: main)
  --region REGION       AWS region (default: us-east-1)
  --set KEY=VALUE       Set Helm values (can be used multiple times)
  --values FILE         Values file to use (can be used multiple times)
  --dry-run             Template only, don't deploy
  --no-wait             Don't wait for deployment to complete
  --timeout DURATION    Timeout for deployment (default: 5m)
  --create-namespace    Create namespace if it doesn't exist
  -h, --help            Show this help message

Examples:
  # Deploy to production
  $0 --chart devops/charts/dashboard-cronjob \\
     --release dashboard-cronjob \\
     --namespace monitoring

  # Deploy to staging with custom values
  $0 --chart devops/charts/dashboard-cronjob \\
     --release dashboard-cronjob-staging \\
     --namespace monitoring-staging \\
     --set schedule="*/30 * * * *" \\
     --set image.tag=test-branch \\
     --create-namespace

  # Dry run (template only)
  $0 --chart devops/charts/dashboard-cronjob \\
     --release dashboard-cronjob \\
     --namespace monitoring \\
     --dry-run

EOF
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --chart)
      CHART_PATH="$2"
      shift 2
      ;;
    --release)
      RELEASE_NAME="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --cluster)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --set)
      EXTRA_ARGS+=(--set "$2")
      shift 2
      ;;
    --values)
      EXTRA_ARGS+=(--values "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-wait)
      WAIT=false
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --create-namespace)
      CREATE_NAMESPACE=true
      shift
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      usage
      ;;
  esac
done

# Validate required parameters
if [[ -z "$CHART_PATH" ]] || [[ -z "$RELEASE_NAME" ]] || [[ -z "$NAMESPACE" ]]; then
  echo -e "${RED}Error: Missing required parameters${NC}"
  usage
fi

# Validate chart path exists
if [[ ! -d "$CHART_PATH" ]]; then
  echo -e "${RED}Error: Chart path does not exist: $CHART_PATH${NC}"
  exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Helm Chart Deployment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Chart:     $CHART_PATH"
echo "Release:   $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Cluster:   $CLUSTER_NAME ($REGION)"
echo "Dry Run:   $DRY_RUN"
echo ""

# Setup kubectl if not dry run
if [[ "$DRY_RUN" == false ]]; then
  echo -e "${GREEN}Setting up kubectl access...${NC}"
  if ! aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION" &> /dev/null; then
    echo -e "${RED}Error: Failed to configure kubectl for cluster: $CLUSTER_NAME${NC}"
    echo "Run: ./devops/k8s/setup-k8s.sh $CLUSTER_NAME $REGION"
    exit 1
  fi
  echo -e "${GREEN}✓ kubectl configured${NC}"
  echo ""

  # Check if namespace exists
  if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${GREEN}✓ Namespace exists: $NAMESPACE${NC}"
  else
    if [[ "$CREATE_NAMESPACE" == true ]]; then
      echo -e "${YELLOW}Creating namespace: $NAMESPACE${NC}"
      kubectl create namespace "$NAMESPACE"
      echo -e "${GREEN}✓ Namespace created${NC}"
    else
      echo -e "${RED}Error: Namespace does not exist: $NAMESPACE${NC}"
      echo "Use --create-namespace to create it automatically"
      exit 1
    fi
  fi
  echo ""
fi

# Build helm command
HELM_CMD=(helm upgrade --install)
HELM_CMD+=("$RELEASE_NAME")
HELM_CMD+=("$CHART_PATH")
HELM_CMD+=(--namespace "$NAMESPACE")

if [[ "$CREATE_NAMESPACE" == true ]]; then
  HELM_CMD+=(--create-namespace)
fi

if [[ "$WAIT" == true ]]; then
  HELM_CMD+=(--wait --timeout "$TIMEOUT")
fi

if [[ "$DRY_RUN" == true ]]; then
  HELM_CMD+=(--dry-run --debug)
fi

# Add extra arguments
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  HELM_CMD+=("${EXTRA_ARGS[@]}")
fi

# Display command
echo -e "${BLUE}Running Helm command:${NC}"
echo "${HELM_CMD[*]}"
echo ""

# Execute helm command
if [[ "$DRY_RUN" == true ]]; then
  echo -e "${YELLOW}═══ DRY RUN - No changes will be made ═══${NC}"
  echo ""
fi

if "${HELM_CMD[@]}"; then
  if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ Deployment successful!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  # Check deployment status"
    echo "  kubectl get all -n $NAMESPACE"
    echo ""
    echo "  # For CronJobs, check the cronjob and recent jobs"
    echo "  kubectl get cronjobs -n $NAMESPACE"
    echo "  kubectl get jobs -n $NAMESPACE --sort-by=.metadata.creationTimestamp"
    echo ""
    echo "  # View logs"
    echo "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$RELEASE_NAME --tail=100"
    echo ""
    echo "  # Manually trigger a cronjob"
    echo "  kubectl create job --from=cronjob/$RELEASE_NAME manual-run-\$(date +%s) -n $NAMESPACE"
    echo ""
  else
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  ✓ Dry run completed - no changes made${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Remove --dry-run to actually deploy"
  fi
else
  echo ""
  echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${RED}  ✗ Deployment failed${NC}"
  echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
  exit 1
fi
