# Configure AWS OIDC for GitHub Actions.

# Documentation:
# https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services

locals {
  audience = "sts.amazonaws.com"
  repo     = "${var.github_org}/${var.github_repo}"
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = [local.audience]
}

# IAM role that GitHub workflows can assume via OIDC.
resource "aws_iam_role" "github_actions" {
  name = "github-actions"

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
            "token.actions.githubusercontent.com:sub" = "repo:${local.repo}:*"
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

resource "github_actions_variable" "aws_role" {
  repository    = var.github_repo
  variable_name = "AWS_ROLE"
  value         = aws_iam_role.github_actions.arn
}
