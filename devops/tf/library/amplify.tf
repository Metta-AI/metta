# ---- Amplify service role ----
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

# ---- Amplify compute role ----
resource "aws_iam_role" "amplify_compute_role" {
  name = "amplify-compute-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Action    = "sts:AssumeRole",
      Principal = { Service = "amplify.amazonaws.com" }
    }]
  })
}

data "aws_iam_policy_document" "amplify_backend_access" {
  statement {
    effect = "Allow"
    actions = [
      "textract:AnalyzeDocument",
      "textract:DetectDocumentText",
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      for bucket in var.s3_buckets : "arn:aws:s3:::${bucket}/*"
    ]
  }
}

resource "aws_iam_policy" "amplify_backend_access" {
  name   = "amplify-next-backend-textract-s3"
  policy = data.aws_iam_policy_document.amplify_backend_access.json
}

resource "aws_iam_role_policy_attachment" "amplify_backend_access_attach" {
  role       = aws_iam_role.amplify_compute_role.name
  policy_arn = aws_iam_policy.amplify_backend_access.arn
}

# ---- Amplify App ----
resource "aws_amplify_app" "library" {
  name       = "softmax-library"
  repository = "https://github.com/Metta-AI/metta.git"

  platform = "WEB_COMPUTE"

  access_token = var.amplify_github_access_token

  iam_service_role_arn = aws_iam_role.amplify_service_role.arn
  compute_role_arn     = aws_iam_role.amplify_compute_role.arn

  # App-level env vars (available to all branches; branch can override)
  environment_variables = merge(local.common_env_vars, local.frontend_env_vars, {
    AMPLIFY_MONOREPO_APP_ROOT = "library"
  })
}

resource "aws_amplify_branch" "main" {
  app_id            = aws_amplify_app.library.id
  branch_name       = "main"
  stage             = "PRODUCTION"
  enable_auto_build = true
}

resource "aws_amplify_domain_association" "domain" {
  app_id      = aws_amplify_app.library.id
  domain_name = var.domain

  sub_domain {
    branch_name = aws_amplify_branch.main.branch_name
    prefix      = ""
  }

  sub_domain {
    branch_name = aws_amplify_branch.main.branch_name
    prefix      = "www"
  }
}
