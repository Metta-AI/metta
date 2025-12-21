# Configure AWS OIDC for GitHub Actions.

# Documentation:
# https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services

locals {
  audience = "sts.amazonaws.com"
  repo     = "${var.github_org}/${var.github_repo}"
  github_oidc_subjects = concat(
    ["repo:${local.repo}:*"],
    [for repo in var.extra_oidc_repos : "repo:${repo}:*"]
  )
}

data "aws_s3_bucket" "softmax_public" {
  bucket = "softmax-public"
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = [local.audience]
}

# IAM role that GitHub workflows can assume via OIDC.
resource "aws_iam_role" "github_actions" {
  name                 = "github-actions"
  max_session_duration = 36000

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = local.audience
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = local.github_oidc_subjects
          }
        }
      }
    ]
  })
}

# Allow GitHub workflows to push to ECR.
resource "aws_iam_role_policy_attachment" "github_ecr_poweruser" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

# Allow GitHub workflows to access the public bucket used across CI tasks.
resource "aws_iam_role_policy" "github_s3_softmax_public" {
  name = "github-actions-softmax-public"
  role = aws_iam_role.github_actions.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
        ]
        Resource = data.aws_s3_bucket.softmax_public.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
          "s3:CreateMultipartUpload",
          "s3:UploadPart",
          "s3:CompleteMultipartUpload",
          "s3:ListMultipartUploadParts",
        ]
        Resource = "${data.aws_s3_bucket.softmax_public.arn}/*"
      }
    ]
  })
}

# This is necessary for github to be able to update the kubeconfig.
resource "aws_iam_role_policy" "github_eks_describe" {
  name = "github-actions-eks-describe"
  role = aws_iam_role.github_actions.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "eks:DescribeCluster"
        Resource = "*"
      }
    ]
  })
}

# Allow GitHub workflows to read Datadog secrets from Secrets Manager.
resource "aws_iam_role_policy" "github_datadog_secrets" {
  name = "github-actions-datadog-secrets"
  role = aws_iam_role.github_actions.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
        ]
        Resource = [
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/api-key-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/app-key-*",
        ]
      }
    ]
  })
}

resource "github_actions_variable" "aws_role" {
  repository    = var.github_repo
  variable_name = "AWS_ROLE"
  value         = aws_iam_role.github_actions.arn
}
