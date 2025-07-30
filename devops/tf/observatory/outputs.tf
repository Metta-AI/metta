output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "postgres_password" {
  value     = random_password.db.result
  sensitive = true
}
