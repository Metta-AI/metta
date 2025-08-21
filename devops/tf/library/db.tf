resource "random_password" "db" {
  length  = 32
  special = true
}

resource "aws_db_instance" "postgres" {
  identifier     = "softmax-library-pg"
  engine         = "postgres"
  engine_version = var.db_postgres_version

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az          = true

  publicly_accessible = true

  db_name  = "softmax_library"
  username = "softmax_library"
  password = random_password.db.result

}
