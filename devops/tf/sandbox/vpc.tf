locals {
  # Single AZ - no need for multi-AZ since each researcher gets one instance
  azs = [data.aws_availability_zones.available.names[0]]

  tags = {
    Terraform   = "true"
    Project     = "alignment-league"
    Environment = var.environment
    Purpose     = "researcher-sandbox"
  }
}

# VPC for researcher sandboxes - isolated from main infrastructure
module "sandbox_vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "sandbox-vpc"
  cidr = var.vpc_cidr

  azs            = local.azs
  public_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k)]

  # Enable internet access for sandboxes
  enable_nat_gateway   = false # Public subnets have direct internet access via IGW
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tag subnets for easy identification
  public_subnet_tags = {
    Name = "sandbox-public"
    Type = "public"
  }

  tags = local.tags
}
