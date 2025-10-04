data "aws_secretsmanager_secret" "oauth_secret" {
  arn = var.oauth_secret_arn
}

data "aws_secretsmanager_secret_version" "oauth_secret" {
  secret_id = data.aws_secretsmanager_secret.oauth_secret.id
}

resource "random_password" "auth_secret" {
  length  = 32
  special = true
}

locals {
  common_env_vars = {
    DATABASE_URL = local.postgres_url
  }

  frontend_env_vars = {
    # Auth Configuration
    NEXTAUTH_SECRET      = random_password.auth_secret.result
    GOOGLE_CLIENT_ID     = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-id"]
    GOOGLE_CLIENT_SECRET = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-secret"]

    # Environment Configuration
    DEV_MODE = "false"
    ALLOWED_EMAIL_DOMAINS = "stem.ai,softmax.com"
  }
}
