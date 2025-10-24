# IRSA role for dashboard cronjob to access Secrets Manager
module "dashboard_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "dashboard-cronjob"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["monitoring:dashboard-cronjob-*"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.dashboard_secrets.arn
  }
}

resource "aws_iam_policy" "dashboard_secrets" {
  name = "dashboard-secrets-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
        ]
        Resource = [
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:github/dashboard-token-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/api-key-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:datadog/app-key-*",
        ]
      }
    ]
  })
}

output "dashboard_irsa_role_arn" {
  value       = module.dashboard_irsa.iam_role_arn
  description = "ARN of the IAM role for dashboard cronjob service account"
}
