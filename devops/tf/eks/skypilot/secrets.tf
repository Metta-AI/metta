data "aws_secretsmanager_secret" "lambda_ai_secret" {
  arn = var.lambda_ai_secret_arn
}

data "aws_secretsmanager_secret_version" "lambda_ai_secret_version" {
  secret_id = data.aws_secretsmanager_secret.lambda_ai_secret.id
}

resource "kubernetes_secret" "lambda_ai_secret" {
  metadata {
    name = "lambda-ai-credentials"
    namespace = "skypilot"
  }

  data = jsondecode(data.aws_secretsmanager_secret_version.lambda_ai_secret_version.secret_string)
}
