# IRSA role for replay mining cronjob to access S3
module "replay_mining_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "replay-mining-cronjob"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["monitoring:replay-mining-cronjob-*"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.replay_mining_s3.arn
  }
}

resource "aws_iam_policy" "replay_mining_s3" {
  name = "replay-mining-s3-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Resource = [
          "arn:aws:s3:::softmax-public",
          "arn:aws:s3:::softmax-public/*",
        ]
      }
    ]
  })
}

output "replay_mining_irsa_role_arn" {
  value       = module.replay_mining_irsa.iam_role_arn
  description = "ARN of the IAM role for replay mining cronjob service account"
}
