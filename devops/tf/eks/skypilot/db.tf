resource "random_password" "db" {
  length  = 32
  special = false
}

# Security group and DB subnet group for accessing the DB from the EKS cluster
# Created by observatory stack (TODO: will be moved to eks stack eventually)
data "aws_security_group" "db" {
  name = "main-postgres-sg"
}

data "aws_db_subnet_group" "db" {
  name = "main-db"
}


resource "aws_db_instance" "postgres" {
  identifier     = "skypilot-pg"
  engine         = "postgres"
  engine_version = var.db_postgres_version

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az          = true

  db_subnet_group_name   = data.aws_db_subnet_group.db.name
  vpc_security_group_ids = [data.aws_security_group.db.id]
  publicly_accessible    = false # stays inside the VPC

  backup_retention_period = 7

  db_name  = "skypilot"
  username = "skypilot"
  password = random_password.db.result

  skip_final_snapshot = true
}

locals {
  postgres_url = "postgresql://${aws_db_instance.postgres.username}:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
}

