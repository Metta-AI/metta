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

# ---- IAM role Amplify will assume to push SSR runtime logs to CloudWatch ----
resource "aws_iam_role" "amplify_service_role" {
  name = "amplify-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Action    = "sts:AssumeRole",
      Principal = { Service = "amplify.amazonaws.com" }
    }]
  })
}

resource "aws_iam_policy" "amplify_cloudwatch_logs" {
  name        = "amplify-cloudwatch-logs"
  description = "Allow Amplify Hosting to create/write CloudWatch Logs for SSR runtime"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:DescribeLogGroups",
        "logs:PutLogEvents"
      ],
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "amplify_logs_attach" {
  role       = aws_iam_role.amplify_service_role.name
  policy_arn = aws_iam_policy.amplify_cloudwatch_logs.arn
}

# ---- Amplify App ----
resource "aws_amplify_app" "library" {
  name       = "softmax-library"
  repository = "https://github.com/Metta-AI/metta.git"

  platform = "WEB_COMPUTE"

  access_token = var.amplify_github_access_token

  iam_service_role_arn = aws_iam_role.amplify_service_role.arn

  # App-level env vars (available to all branches; branch can override)
  environment_variables = {
    AMPLIFY_MONOREPO_APP_ROOT = "library"
    DATABASE_URL              = "postgresql://${aws_db_instance.postgres.username}:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
    DEV_MODE                  = "false"

    # Auth
    AUTH_SECRET          = random_password.auth_secret.result
    GOOGLE_CLIENT_ID     = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-id"]
    GOOGLE_CLIENT_SECRET = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret.secret_string)["client-secret"]

    # Asana Configuration
    ASANA_API_KEY           = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ASANA_API_KEY"]
    ASANA_PAPERS_PROJECT_ID = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ASANA_PAPERS_PROJECT_ID"]

    # # Adobe PDF Services Configuration
    ADOBE_CLIENT_ID         = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ADOBE_CLIENT_ID"]
    ADOBE_CLIENT_SECRET     = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ADOBE_CLIENT_SECRET"]
    USE_LLM_ADOBE_SELECTION = "true"

    ANTHROPIC_API_KEY = jsondecode(data.aws_secretsmanager_secret_version.library_secrets.secret_string)["ANTHROPIC_API_KEY"]
  }
}

resource "aws_amplify_branch" "main" {
  app_id            = aws_amplify_app.library.id
  branch_name       = "library-deployment" # TODO: change to main
  stage             = "PRODUCTION"
  enable_auto_build = true
}

resource "aws_amplify_domain_association" "domain" {
  app_id      = aws_amplify_app.library.id
  domain_name = "library.softmax-research.net"

  sub_domain {
    branch_name = aws_amplify_branch.main.branch_name
    prefix      = ""
  }

  sub_domain {
    branch_name = aws_amplify_branch.main.branch_name
    prefix      = "www"
  }
}
