locals {
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)

  tags = {
    Terraform = "true"
  }

  # Add new roles here to grant them access to the EKS cluster.
  admins = [
    "arn:aws:iam::751442549699:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_AdministratorAccess_ac2ae6482eae17c4",
    "arn:aws:iam::751442549699:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_PowerUserAccess_765d58f6b3d9465f",
    data.aws_iam_role.github_actions.arn
  ]
}

data "aws_iam_role" "github_actions" {
  name = "github-actions"
}

data "aws_availability_zones" "available" {
  # Exclude local zones
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}


################################################################################
# EKS Module
################################################################################

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.31"

  cluster_name                   = var.cluster_name
  cluster_version                = var.cluster_version
  cluster_endpoint_public_access = true

  enable_cluster_creator_admin_permissions = true

  # EKS auto mode - AWS will scale the node group based on the workload
  cluster_compute_config = {
    enabled    = true
    node_pools = ["general-purpose"]
  }

  # https://www.reddit.com/r/Terraform/comments/znomk4/ebs_csi_driver_entirely_from_terraform_on_aws_eks/
  cluster_addons = {
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  node_iam_role_additional_policies = {
    AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets


  tags = local.tags
}

################################################################################
# Supporting Resources
################################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = var.cluster_name
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 52)]

  enable_nat_gateway = true
  single_nat_gateway = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

resource "aws_eks_access_entry" "admin" {
  for_each      = toset(local.admins)
  cluster_name  = module.eks.cluster_name
  principal_arn = each.value
}

resource "aws_eks_access_policy_association" "admin" {
  for_each      = toset(local.admins)
  cluster_name  = module.eks.cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = each.value

  access_scope {
    type = "cluster"
  }
}
