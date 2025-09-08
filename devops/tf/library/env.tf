data "aws_secretsmanager_secret" "oauth_secret" {
  arn = var.oauth_secret_arn
}

data "aws_secretsmanager_secret_version" "oauth_secret" {
  secret_id = data.aws_secretsmanager_secret.oauth_secret.id
}

data "aws_secretsmanager_secret" "library_secrets" {
  arn = var.library_secrets_arn
}

data "aws_secretsmanager_secret_version" "library_secrets" {
  secret_id = data.aws_secretsmanager_secret.library_secrets.id
}

resource "random_password" "auth_secret" {
  length  = 32
  special = true
}

locals {
  common_env_vars = {
    S3_BUCKET    = var.main_s3_bucket
    DATABASE_URL = local.postgres_url

    # Asana Configuration
    ASANA_API_KEY           = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ASANA_API_KEY"]
    ASANA_PAPERS_PROJECT_ID = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ASANA_PAPERS_PROJECT_ID"]

    # # Adobe PDF Services Configuration
    ADOBE_CLIENT_ID         = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ADOBE_CLIENT_ID"]
    ADOBE_CLIENT_SECRET     = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ADOBE_CLIENT_SECRET"]
    USE_LLM_ADOBE_SELECTION = "true"

    ANTHROPIC_API_KEY = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ANTHROPIC_API_KEY"]
  }

  frontend_env_vars = {
    DEV_MODE = "false"

    # Auth
    AUTH_SECRET          = random_password.auth_secret.result
    NEXTAUTH_URL         = "https://${var.domain}"
    GOOGLE_CLIENT_ID     = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-id"]
    GOOGLE_CLIENT_SECRET = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-secret"]
  }
}
