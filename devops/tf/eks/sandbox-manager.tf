# IRSA role for sandbox-manager service to manage EC2 instances
module "sandbox_manager_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.34"

  role_name = "sandbox-manager"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["default:sandbox-manager"]
    }
  }

  role_policy_arns = {
    policy = aws_iam_policy.sandbox_manager.arn
  }
}

# IAM policy for sandbox manager service
# This policy allows the FastAPI backend to manage EC2 instances
resource "aws_iam_policy" "sandbox_manager" {
  name        = "sandbox-manager-policy"
  description = "Allows sandbox manager service to orchestrate EC2 instances for researchers"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # EC2 instance management
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:TerminateInstances",
          "ec2:RebootInstances",
          "ec2:ModifyInstanceAttribute"
        ]
        Resource = [
          "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:instance/*",
          "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:volume/*",
          "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:network-interface/*",
          "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:subnet/*",
          "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:security-group/*"
        ]
      },
      # Allow RunInstances to use AMIs
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances"
        ]
        Resource = "arn:aws:ec2:*::image/ami-*"
      },
      # EC2 tagging (for cost tracking)
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ]
        Resource = "arn:aws:ec2:*:${data.aws_caller_identity.current.account_id}:*"
      },
      # EC2 describe operations (for status checks)
      {
        Effect = "Allow"
        Action = [
          "ec2:Describe*"
        ]
        Resource = "*"
      },
      # Pass IAM role to EC2 instances
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/sandbox-instance-role"
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "ec2.amazonaws.com"
          }
        }
      },
      # Get instance profile (needed for RunInstances)
      {
        Effect = "Allow"
        Action = [
          "iam:GetInstanceProfile"
        ]
        Resource = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:instance-profile/sandbox-instance-profile"
      },
      # Cost Explorer for spending tracking
      {
        Effect = "Allow"
        Action = [
          "ce:GetCostAndUsage",
          "ce:GetCostForecast"
        ]
        Resource = "*"
      },
      # CloudWatch for metrics (optional, for monitoring)
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics"
        ]
        Resource = "*"
      }
    ]
  })
}

# Output the role ARN for reference
output "sandbox_manager_irsa_role_arn" {
  value       = module.sandbox_manager_irsa.iam_role_arn
  description = "ARN of the IAM role for sandbox-manager service account"
}
