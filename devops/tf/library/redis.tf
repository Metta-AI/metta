data "aws_eks_cluster" "eks" {
  name = var.eks_cluster_name
}

locals {
  vpc_id    = data.aws_eks_cluster.eks.vpc_config[0].vpc_id
  eks_sg_id = data.aws_eks_cluster.eks.vpc_config[0].cluster_security_group_id
}

data "aws_vpc" "eks_vpc" {
  id = local.vpc_id
}

data "aws_subnets" "eks_private" {
  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
  filter {
    # Kubernetes adds this tag to the private subnets it wants for
    # internal load balancers; perfect for RDS.
    name   = "tag:kubernetes.io/role/internal-elb"
    values = ["1"]
  }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.eks_cluster_name}-library-redis-subnets"
  subnet_ids = data.aws_subnets.eks_private.ids
}

resource "aws_security_group" "redis" {
  name   = "${var.eks_cluster_name}-library-redis-sg"
  vpc_id = local.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [local.eks_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_elasticache_parameter_group" "redis" {
  name        = "softmax-library-redis-params"
  family      = "redis7"
  description = "Parameter group for softmax-library Redis"

  parameter {
    name  = "maxmemory-policy"
    value = "noeviction"
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id = "softmax-library-redis"
  description          = "Redis for softmax-library job queue"
  engine               = "redis"
  node_type            = "cache.t4g.micro"
  num_cache_clusters   = 1

  # encryption is required when using auth token
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result

  parameter_group_name = aws_elasticache_parameter_group.redis.name
  apply_immediately    = true

  security_group_ids = [aws_security_group.redis.id]
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
}
