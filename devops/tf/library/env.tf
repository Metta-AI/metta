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

resource "random_password" "redis_password" {
  length  = 32
  special = false
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

    REDIS_HOST     = aws_elasticache_replication_group.main.primary_endpoint_address
    REDIS_PASSWORD = random_password.redis_password.result
    REDIS_TLS      = "true"

    # AWS SES Configuration for Email Notifications
    SES_SMTP_HOST     = local.ses_smtp_endpoint
    SES_SMTP_PORT     = "587"
    SES_SMTP_USERNAME = local.ses_smtp_username
    SES_SMTP_PASSWORD = local.ses_smtp_password
    SES_FROM_EMAIL    = local.ses_from_email
    SES_FROM_NAME     = var.ses_from_name
    SES_REGION        = var.region
  }

  frontend_env_vars = {
    # Auth
    AUTH_SECRET          = random_password.auth_secret.result
    GOOGLE_CLIENT_ID     = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-id"]
    GOOGLE_CLIENT_SECRET = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-secret"]
  }
}
