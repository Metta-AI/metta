data "aws_eks_cluster" "eks" {
  name = var.eks_cluster_name
}

# Workers usually share the *cluster* security group; that is enough
# to scope DB access to the nodes.
locals {
  vpc_id    = data.aws_eks_cluster.eks.vpc_config[0].vpc_id
  eks_sg_id = data.aws_eks_cluster.eks.vpc_config[0].cluster_security_group_id
}

data "aws_subnets" "private" {
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

resource "aws_db_subnet_group" "this" {
  name       = "${var.eks_cluster_name}-db"
  subnet_ids = data.aws_subnets.private.ids
}

resource "aws_security_group" "db" {
  name        = "${var.eks_cluster_name}-postgres-sg"
  description = "Allow Postgres from EKS worker nodes"
  vpc_id      = local.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [local.eks_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # let the DB reach out for backups, etc.
  }
}

resource "random_password" "db" {
  length  = 20
  special = true
}

resource "aws_db_instance" "postgres" {
  identifier     = "${var.eks_cluster_name}-pg"
  engine         = "postgres"
  engine_version = var.db_postgres_version

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az          = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false # stays inside the VPC

  db_name  = "metta"
  username = "metta"
  password = random_password.db.result

}
