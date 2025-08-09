# IRSA role for orchestrator and eval workers to access S3
module "orchestrator_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "orchestrator-eval-worker"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["orchestrator:orchestrator-orchestrator"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.orchestrator_s3.arn
  }
}

resource "aws_iam_policy" "orchestrator_s3" {
  name = "orchestrator-s3-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
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

# Output the role ARN for reference
output "orchestrator_irsa_role_arn" {
  value = module.orchestrator_irsa.iam_role_arn
  description = "ARN of the IAM role for orchestrator service account"
}
