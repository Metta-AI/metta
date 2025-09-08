data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default_vpc" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "softmax-library-redis-subnets"
  subnet_ids = data.aws_subnets.default_vpc.ids
}

resource "aws_security_group" "redis_public" {
  name   = "softmax-library-redis-sg"
  vpc_id = data.aws_vpc.default.id

  # TEMP: open to world. Required only if a public entrypoint (e.g., NLB) fronts Redis.
  # ElastiCache itself is not publicly accessible.
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id = "softmax-library-redis"
  description          = "Redis for softmax-library job job queue"
  engine               = "redis"
  node_type            = "cache.t4g.micro"
  num_cache_clusters   = 1

  auth_token = random_password.redis_password.result

  security_group_ids = [aws_security_group.redis_public.id]
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
}
