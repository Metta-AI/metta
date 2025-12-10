resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

data "aws_secretsmanager_secret" "anthropic_api_key" {
  name = "anthropic/api-key"
}

data "aws_secretsmanager_secret_version" "anthropic_api_key_version" {
  secret_id = data.aws_secretsmanager_secret.anthropic_api_key.id
}

resource "random_password" "auth_secret" {
  length  = 27
  special = false
}

resource "kubernetes_secret" "observatory_backend_env" {
  metadata {
    name      = "observatory-backend-env"
    namespace = kubernetes_namespace.observatory.metadata[0].name
  }
  data = {
    STATS_DB_URI = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
    # used by SQL query generator
    ANTHROPIC_API_KEY = data.aws_secretsmanager_secret_version.anthropic_api_key_version.secret_string
    # bypass for token auth in app_backend, allows softmax.com -> observatory communication
    OBSERVATORY_AUTH_SECRET = random_password.auth_secret.result
  }
}
