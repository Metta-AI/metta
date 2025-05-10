provider "aws" {
  region = "us-east-1"
}

# -------------------------------------------------------------------
# 1) Networking: VPC + 2 Public Subnets + IGW + Route Table
# -------------------------------------------------------------------

data "aws_availability_zones" "azs" {
  state = "available"
}

resource "aws_vpc" "eks" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = "eks-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.eks.id
  tags   = { Name = "eks-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.eks.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = { Name = "eks-public-rt" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.eks.id
  cidr_block              = cidrsubnet(aws_vpc.eks.cidr_block, 8, count.index)
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.azs.names[count.index]
  tags = {
    Name = "eks-public-${count.index}"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# -------------------------------------------------------------------
# 2) EKS Control‐Plane IAM Role
# -------------------------------------------------------------------

data "aws_iam_policy_document" "eks_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_cluster" {
  name               = "eks-cluster-role"
  assume_role_policy = data.aws_iam_policy_document.eks_assume.json
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# -------------------------------------------------------------------
# 3) EKS Cluster
# -------------------------------------------------------------------

resource "aws_eks_cluster" "main" {
  # Called "main" but for now it's going to be used only for skypilot-api-server.
  # We're deploying ML jobs to EC2 with skypilot, no k8s.
  name     = "main-eks"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.32"

  vpc_config {
    subnet_ids             = aws_subnet.public[*].id
    endpoint_public_access = true
    public_access_cidrs    = ["0.0.0.0/0"]
  }

  tags = { Name = "main-eks" }
}

# -------------------------------------------------------------------
# 4) Node-Group IAM Role
# -------------------------------------------------------------------

data "aws_iam_policy_document" "node_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_node" {
  name               = "eks-nodegroup-role"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
}

# Attach the three standard policies:
resource "aws_iam_role_policy_attachment" "node_worker" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "node_cni" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "node_ecr" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# -------------------------------------------------------------------
# 5) Managed Node Group
# -------------------------------------------------------------------

resource "aws_eks_node_group" "main_ng" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "main-ng"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.public[*].id

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 1
  }

  instance_types = ["c5.2xlarge"]
  capacity_type  = "ON_DEMAND"
}

# -------------------------------------------------------------------
# 6) Outputs
# -------------------------------------------------------------------

output "cluster_name" {
  value = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}

output "cluster_ca_data" {
  value = aws_eks_cluster.main.certificate_authority[0].data
}

# Allow admin access

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "eks_assume_admin" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/AWSReservedSSO_AdministratorAccess_*"]
    }
  }
}

resource "aws_iam_role" "eks_cluster_admin" {
  name               = "eks-cluster-admin-role"
  assume_role_policy = data.aws_iam_policy_document.eks_assume.json
}
