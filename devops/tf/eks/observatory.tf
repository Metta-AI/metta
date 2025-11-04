# IRSA role for observatory and eval workers to access S3
module "observatory_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "observatory-backend"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["observatory:observatory-backend"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.observatory_s3.arn
  }
}

resource "aws_iam_policy" "observatory_s3" {
  name = "observatory-s3-access"
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
          "arn:aws:s3:::observatory-private",
          "arn:aws:s3:::observatory-private/*",
        ]
      }
    ]
  })
}

# Output the role ARN for reference
output "observatory_irsa_role_arn" {
  value = module.observatory_irsa.iam_role_arn
  description = "ARN of the IAM role for observatory service account"
}
