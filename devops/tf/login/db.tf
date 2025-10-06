resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_security_group" "rds_public" {
  name = "softmax-login-pg-sg"

  # TEMP: open to world. Necessary for development (unless we configure lambda proxy).
  ingress {
    from_port   = 5432
    to_port     = 5432
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

resource "aws_db_instance" "postgres" {
  identifier     = "softmax-login-pg"
  engine         = "postgres"
  engine_version = var.db_postgres_version

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az          = true

  publicly_accessible    = true
  vpc_security_group_ids = [aws_security_group.rds_public.id]

  backup_retention_period = 7

  db_name  = "softmax_login"
  username = "softmax_login"
  password = random_password.db.result
}

locals {
  postgres_url = "postgresql://${aws_db_instance.postgres.username}:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
}