# IRSA role for webhook service to access Secrets Manager
module "webhook_service_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "webhook-service"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["observatory:webhook-service"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.webhook_service_secrets.arn
  }
}

resource "aws_iam_policy" "webhook_service_secrets" {
  name = "webhook-service-secrets-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:github/webhook-secret-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/access-token-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/api-key-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/atlas_app-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:github/token-*"
        ]
      }
    ]
  })
}

# Output the role ARN for reference
output "webhook_service_irsa_role_arn" {
  value       = module.webhook_service_irsa.iam_role_arn
  description = "ARN of the IAM role for webhook service account"
}

