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
          "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:instance/*",
          "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:volume/*",
          "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:network-interface/*",
          "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:subnet/*",
          "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:security-group/*"
        ]
        Condition = {
          StringEquals = {
            "aws:RequestedRegion" = var.region
          }
        }
      },
      # Allow RunInstances to use AMIs
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances"
        ]
        Resource = "arn:aws:ec2:${var.region}::image/ami-*"
      },
      # EC2 tagging (for cost tracking)
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ]
        Resource = "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:*"
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
        Resource = aws_iam_role.sandbox_instance.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "ec2.amazonaws.com"
          }
        }
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

# IAM user for sandbox manager service (runs in EKS)
resource "aws_iam_user" "sandbox_manager" {
  name = "sandbox-manager"
  path = "/service/"

  tags = merge(local.tags, {
    Name    = "sandbox-manager"
    Purpose = "Service account for sandbox orchestration"
  })
}

# Attach policy to user
resource "aws_iam_user_policy_attachment" "sandbox_manager" {
  user       = aws_iam_user.sandbox_manager.name
  policy_arn = aws_iam_policy.sandbox_manager.arn
}

# Create access key for sandbox manager (to be stored in k8s secret)
resource "aws_iam_access_key" "sandbox_manager" {
  user = aws_iam_user.sandbox_manager.name
}

# IAM role for sandbox EC2 instances
# This role is assumed by instances and grants them necessary permissions
resource "aws_iam_role" "sandbox_instance" {
  name        = "sandbox-instance-role"
  description = "IAM role for researcher sandbox EC2 instances"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.tags, {
    Name = "sandbox-instance-role"
  })
}

# Policy for sandbox instances - minimal permissions
resource "aws_iam_policy" "sandbox_instance" {
  name        = "sandbox-instance-policy"
  description = "Minimal permissions for researcher sandbox EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 read access for datasets (if needed)
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::softmax-*",
          "arn:aws:s3:::softmax-*/*"
        ]
      },
      # S3 write access for outputs (scoped to user-specific prefix)
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::softmax-sandbox-outputs/*"
        ]
      },
      # CloudWatch Logs (for debugging)
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/ec2/sandbox/*"
      },
      # ECR read-only (for pulling Docker images)
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.tags
}

# Attach policy to instance role
resource "aws_iam_role_policy_attachment" "sandbox_instance" {
  role       = aws_iam_role.sandbox_instance.name
  policy_arn = aws_iam_policy.sandbox_instance.arn
}

# Create instance profile (required to attach IAM role to EC2)
resource "aws_iam_instance_profile" "sandbox" {
  name = "sandbox-instance-profile"
  role = aws_iam_role.sandbox_instance.name

  tags = merge(local.tags, {
    Name = "sandbox-instance-profile"
  })
}
