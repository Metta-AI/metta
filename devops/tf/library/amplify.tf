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


resource "aws_amplify_app" "library" {
  name       = "softmax-library"
  repository = "https://github.com/Metta-AI/metta.git"

  platform = "WEB_COMPUTE"

  access_token = var.amplify_github_access_token

  # App-level env vars (available to all branches; branch can override)
  environment_variables = {
    AMPLIFY_MONOREPO_APP_ROOT = "library"
    DATABASE_URL              = "postgresql://${aws_db_instance.postgres.username}:${random_password.db.result}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
    DEV_MODE                  = "false"

    # Google OAuth Configuration (for production authentication)
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

  build_spec = <<-EOT
version: 1
applications:
  - appRoot: library
    frontend:
      phases:
        preBuild:
          commands: ['corepack enable', 'pnpm install', 'pnpm run db:generate']
        build:
          commands: ['pnpm run build']
      artifacts:
        baseDirectory: library/.next
        files:
          - '**/*'
      cache:
        paths:
          - 'library/.next/cache/**/*'
          - 'node_modules/**/*'
          - 'library/node_modules/**/*'
  EOT
}

resource "aws_amplify_branch" "main" {
  app_id            = aws_amplify_app.library.id
  branch_name       = "main"
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
