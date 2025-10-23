#!/usr/bin/env bash
# Setup kubectl and helm for EKS cluster access

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CLUSTER_NAME="${1:-main}"
REGION="${2:-us-east-1}"

echo -e "${GREEN}Setting up Kubernetes access...${NC}"
echo "Cluster: $CLUSTER_NAME"
echo "Region: $REGION"
echo ""

# Check prerequisites
check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo -e "${RED}Error: $1 is not installed${NC}"
    echo "Please install $1 and try again"
    exit 1
  fi
}

echo "Checking prerequisites..."
check_command aws
check_command kubectl
check_command helm
echo -e "${GREEN}✓ All prerequisites installed${NC}"
echo ""

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
  echo -e "${RED}Error: AWS credentials not configured${NC}"
  echo "Run: aws configure"
  exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_USER=$(aws sts get-caller-identity --query Arn --output text | cut -d'/' -f2)
echo -e "${GREEN}✓ AWS credentials found${NC}"
echo "  Account: $AWS_ACCOUNT"
echo "  User: $AWS_USER"
echo ""

# Update kubeconfig
echo "Updating kubeconfig for EKS cluster..."
if aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"; then
  echo -e "${GREEN}✓ Kubeconfig updated${NC}"
else
  echo -e "${RED}Error: Failed to update kubeconfig${NC}"
  echo "Make sure you have access to the EKS cluster: $CLUSTER_NAME"
  exit 1
fi
echo ""

# Verify kubectl access
echo "Verifying kubectl access..."
if kubectl cluster-info &> /dev/null; then
  echo -e "${GREEN}✓ kubectl access verified${NC}"
  kubectl cluster-info | head -1
else
  echo -e "${RED}Error: Cannot access cluster${NC}"
  exit 1
fi
echo ""

# List available namespaces
echo "Available namespaces:"
kubectl get namespaces -o custom-columns=NAME:.metadata.name,STATUS:.status.phase --no-headers \
  | grep -E "(monitoring|observatory|skypilot)" || echo "  (none found)"
echo ""

# Check helm
echo "Helm version:"
helm version --short
echo ""

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "You can now use kubectl and helm commands:"
echo "  kubectl get pods -n monitoring"
echo "  helm list -n monitoring"
echo ""
echo "To deploy the dashboard cronjob:"
echo "  ./devops/k8s/deploy-helm-chart.sh \\"
echo "    --chart devops/charts/dashboard-cronjob \\"
echo "    --release dashboard-cronjob \\"
echo "    --namespace monitoring \\"
echo "    --cluster $CLUSTER_NAME"
