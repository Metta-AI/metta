data "aws_caller_identity" "current" {}

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

  tags = local.tags
}

# Policy for sandbox instances - minimal permissions
resource "aws_iam_policy" "sandbox_instance" {
  name        = "sandbox-instance-policy"
  description = "Minimal permissions for researcher sandbox EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 write access for outputs
      # Researchers can write anywhere in sandbox-outputs bucket
      # FastAPI will enforce user_id prefix at application level
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::softmax-sandbox-outputs",
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
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws/ec2/sandbox/*"
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
