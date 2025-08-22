output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "postgres_url" {
  value = nonsensitive(local.postgres_url)
}
