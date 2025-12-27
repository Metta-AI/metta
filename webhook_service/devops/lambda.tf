# Lambda function for GitHub webhook service

data "aws_secretsmanager_secret" "github_webhook_secret" {
  name = "github/webhook-secret"
}

data "aws_secretsmanager_secret_version" "github_webhook_secret_version" {
  secret_id = data.aws_secretsmanager_secret.github_webhook_secret.id
}

data "aws_secretsmanager_secret" "asana_access_token" {
  name = "asana/access-token"
}

data "aws_secretsmanager_secret_version" "asana_access_token_version" {
  secret_id = data.aws_secretsmanager_secret.asana_access_token.id
}

data "aws_secretsmanager_secret" "asana_workspace_gid" {
  name = "asana/workspace-gid"
}

data "aws_secretsmanager_secret_version" "asana_workspace_gid_version" {
  secret_id = data.aws_secretsmanager_secret.asana_workspace_gid.id
}

data "aws_secretsmanager_secret" "asana_project_gid" {
  name = "asana/bugs-project-gid"
}

data "aws_secretsmanager_secret_version" "asana_project_gid_version" {
  secret_id = data.aws_secretsmanager_secret.asana_project_gid.id
}

resource "aws_iam_role" "webhook_lambda_role" {
  name = "github-webhook-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "webhook_lambda_policy" {
  name = "github-webhook-lambda-policy"
  role = aws_iam_role.webhook_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          data.aws_secretsmanager_secret.github_webhook_secret.arn,
          data.aws_secretsmanager_secret.asana_access_token.arn,
          data.aws_secretsmanager_secret.asana_workspace_gid.arn,
          data.aws_secretsmanager_secret.asana_project_gid.arn
        ]
      }
    ]
  })
}

resource "aws_lambda_function" "webhook_service" {
  filename         = "webhook_service.zip"
  function_name    = "github-webhook-service"
  role            = aws_iam_role.webhook_lambda_role.arn
  handler         = "lambda_function.handler"
  runtime         = "python3.12"
  timeout         = 30
  memory_size     = 256

  environment {
    variables = {
      USE_AWS_SECRETS = "true"
      AWS_REGION      = "us-east-1"
    }
  }

  source_code_hash = filebase64sha256("webhook_service.zip")
}

resource "aws_lambda_function_url" "webhook_service_url" {
  function_name      = aws_lambda_function.webhook_service.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_headers     = ["*"]
    allow_methods     = ["POST", "GET"]
    allow_origins     = ["*"]
    expose_headers    = []
    max_age           = 0
  }
}

output "webhook_service_url" {
  value = aws_lambda_function_url.webhook_service_url.function_url
}


