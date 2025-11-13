# IAM role for sandbox-manager service to manage EC2 instances
# Uses EKS Pod Identity (not IRSA) for easier configuration
resource "aws_iam_role" "sandbox_manager" {
  name = "sandbox-manager"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEksAuthToAssumeRoleForPodIdentity"
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })

  tags = local.tags
}

# IAM policy for sandbox manager service
resource "aws_iam_policy" "sandbox_manager" {
  name        = "sandbox-manager-policy"
  description = "Allows sandbox manager service to orchestrate EC2 instances for researchers"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # EC2 instance management (scoped to us-east-1 for security)
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
          "arn:aws:ec2:us-east-1:751442549699:instance/*",
          "arn:aws:ec2:us-east-1:751442549699:volume/*",
          "arn:aws:ec2:us-east-1:751442549699:network-interface/*",
          "arn:aws:ec2:us-east-1:751442549699:subnet/*",
          "arn:aws:ec2:us-east-1:751442549699:security-group/*",
          "arn:aws:ec2:us-east-1:751442549699:key-pair/*"
        ]
      },
      # Allow RunInstances to use AMIs (any public or account AMI in us-east-1)
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances"
        ]
        Resource = "arn:aws:ec2:us-east-1::image/ami-*"
      },
      # EC2 tagging (for cost tracking)
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ]
        Resource = "arn:aws:ec2:us-east-1:751442549699:*"
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
        Resource = "arn:aws:iam::751442549699:role/sandbox-instance-role"
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
        Resource = "arn:aws:iam::751442549699:instance-profile/sandbox-instance-profile"
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

  tags = local.tags
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "sandbox_manager" {
  role       = aws_iam_role.sandbox_manager.name
  policy_arn = aws_iam_policy.sandbox_manager.arn
}

# Associate role with EKS service account using Pod Identity
resource "aws_eks_pod_identity_association" "sandbox_manager" {
  cluster_name    = data.aws_eks_cluster.main.name
  namespace       = "default"
  service_account = "sandbox-manager"
  role_arn        = aws_iam_role.sandbox_manager.arn
}

# Output the role ARN for reference
output "sandbox_manager_role_arn" {
  value       = aws_iam_role.sandbox_manager.arn
  description = "ARN of the IAM role for sandbox-manager service account"
}
