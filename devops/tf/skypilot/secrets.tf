data "aws_secretsmanager_secret" "lambda_ai_secret" {
  name = "LambdaAI"
}

data "aws_secretsmanager_secret_version" "lambda_ai_secret_version" {
  secret_id = data.aws_secretsmanager_secret.lambda_ai_secret.id
}

# Deprecated - replace with lambda-credentials after skypilot-values.yaml is updated
resource "kubernetes_secret_v1" "lambda_ai_secret" {
  metadata {
    name      = "lambda-ai-credentials"
    namespace = "skypilot"
  }

  data = jsondecode(data.aws_secretsmanager_secret_version.lambda_ai_secret_version.secret_string)
}

resource "kubernetes_secret_v1" "lambda_credentials" {
  metadata {
    name      = "lambda-credentials" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = jsondecode(data.aws_secretsmanager_secret_version.lambda_ai_secret_version.secret_string)
}

resource "kubernetes_secret_v1" "skypilot_api_server_credentials" {
  metadata {
    name      = "aws-credentials" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    aws_access_key_id     = aws_iam_access_key.skypilot_api_server.id
    aws_secret_access_key = aws_iam_access_key.skypilot_api_server.secret
  }
}

resource "kubernetes_secret_v1" "skypilot_db_connection" {
  metadata {
    name      = "skypilot-db-connection-uri"
    namespace = "skypilot"
  }

  data = {
    connection_string = local.postgres_url
  }
}
