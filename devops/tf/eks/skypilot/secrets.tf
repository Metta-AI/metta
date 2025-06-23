data "aws_secretsmanager_secret" "lambda_ai_secret" {
  arn = var.lambda_ai_secret_arn
}

data "aws_secretsmanager_secret_version" "lambda_ai_secret_version" {
  secret_id = data.aws_secretsmanager_secret.lambda_ai_secret.id
}

resource "kubernetes_secret" "lambda_ai_secret" {
  metadata {
    name      = "lambda-ai-credentials"
    namespace = "skypilot"
  }

  data = jsondecode(data.aws_secretsmanager_secret_version.lambda_ai_secret_version.secret_string)
}

resource "random_password" "skypilot_password" {
  length  = 40
  special = false
}

resource "kubernetes_secret" "skypilot_auth" {
  metadata {
    name      = "skypilot-basic-auth" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    auth = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }
}

resource "kubernetes_secret" "skypilot_api_server_credentials" {
  metadata {
    name      = "aws-credentials" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    aws_access_key_id     = aws_iam_access_key.skypilot_api_server.id
    aws_secret_access_key = aws_iam_access_key.skypilot_api_server.secret
  }
}
