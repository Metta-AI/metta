# GitHub webhook service Lambda function
# Deployed via Spacelift - see devops/tf/README.md for deployment workflow

variable "asana_workspace_gid" {
  type        = string
  description = "Asana workspace GID (not a secret)"
}

variable "asana_project_gid" {
  type        = string
  description = "Asana project GID for bugs (not a secret)"
}

# Build Lambda package zip file
resource "null_resource" "build_lambda_package" {
  triggers = {
    # Rebuild when source code or dependencies change
    source_hash = sha256(join("", [
      for f in fileset("${path.module}/../src", "**/*.py") : filesha256("${path.module}/../src/${f}")
    ]))
    pyproject_hash = filesha256("${path.module}/../pyproject.toml")
    lambda_handler_hash = filesha256("${path.module}/../lambda_function.py")
  }

  provisioner "local-exec" {
    command = <<-EOT
      cd ${path.module}/.. && \
      rm -rf .lambda_package ${path.module}/webhook_service.zip && \
      mkdir -p .lambda_package && \
      python3 -m pip install --target .lambda_package asana boto3 fastapi httpx mangum pydantic pydantic-settings requests 'uvicorn[standard]' && \
      cp -r src .lambda_package/ && \
      cp lambda_function.py .lambda_package/ && \
      cd .lambda_package && \
      zip -r ${path.module}/webhook_service.zip . -q && \
      cd .. && \
      rm -rf .lambda_package
    EOT
  }
}

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
          data.aws_secretsmanager_secret.asana_access_token.arn
        ]
      }
    ]
  })
}

resource "aws_lambda_function" "webhook_service" {
  depends_on = [null_resource.build_lambda_package]

  filename         = "${path.module}/webhook_service.zip"
  function_name    = "github-webhook-service"
  role            = aws_iam_role.webhook_lambda_role.arn
  handler         = "lambda_function.handler"
  runtime         = "python3.12"
  timeout         = 30
  memory_size     = 256

  environment {
    variables = {
      USE_AWS_SECRETS   = "true"
      AWS_REGION        = "us-east-1"
      ASANA_WORKSPACE_GID = var.asana_workspace_gid
      ASANA_PROJECT_GID   = var.asana_project_gid
    }
  }

  source_code_hash = filebase64sha256("${path.module}/webhook_service.zip")
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


