output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "postgres_password" {
  value     = random_password.db.result
  sensitive = true
}

output "postgres_url" {
  value     = "postgresql://${aws_db_instance.postgres.username}:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  sensitive = true
}
