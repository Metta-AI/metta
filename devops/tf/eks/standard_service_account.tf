# IRSA role for standard service account used by cronjobs
module "standard_service_account_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "standard-service-account"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = [
        "monitoring:standard-service-account",
      ]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.standard_service_account.arn
  }
}

resource "aws_iam_policy" "standard_service_account" {
  name = "standard-service-account-access"
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
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/access-token-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/workspace-gid-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:asana/bugs-project-gid-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:wandb/api-key-*",
          "arn:aws:secretsmanager:us-east-1:751442549699:secret:GEMINI-API-KEY*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeVolumes",
          "ec2:DescribeSnapshots",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:CreateMultipartUpload",
          "s3:UploadPart",
          "s3:CompleteMultipartUpload",
          "s3:AbortMultipartUpload",
        ]
        Resource = "arn:aws:s3:::softmax-public/*"
      }
    ]
  })
}

output "standard_service_account_irsa_role_arn" {
  value       = module.standard_service_account_irsa.iam_role_arn
  description = "ARN of the IAM role for the standard cronjob service account"
}
