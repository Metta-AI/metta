data "aws_caller_identity" "current" {}

locals {
  aws_integration_name = "softmax-aws"
  iam_role_name        = "Spacelift"
  # AWS account by spacelift, fixed
  # (see spacelift docs, https://docs.spacelift.io/integrations/cloud-providers/aws#configure-trust-policy)
  spacelift_aws_id = "324880187172"
  # Our account at spacelift
  spacelift_account = "Metta-AI"
}

# Create IAM role for Spacelift to assume
resource "aws_iam_role" "spacelift" {
  name        = local.iam_role_name
  description = "Role allowing Spacelift to deploy to AWS. Created by Terraform."

  assume_role_policy = jsonencode({
    # https://docs.spacelift.io/integrations/cloud-providers/aws#configure-trust-policy
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          "AWS" : "arn:aws:iam::${local.spacelift_aws_id}:root"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringLike = {
            "sts:ExternalId" : "${local.spacelift_account}@*"
          }
        }
      }
    ]
  })
}

# Make the role a poweruser
resource "aws_iam_role_policy_attachment" "spacelift_poweruser" {
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
  role       = aws_iam_role.spacelift.name
}


# Register the integration with Spacelift
# (Will be available at https://metta-ai.app.spacelift.io/cloud-integrations)
resource "spacelift_aws_integration" "softmax" {
  name             = local.aws_integration_name
  role_arn         = aws_iam_role.spacelift.arn
  duration_seconds = 3600 # default is 15 min, which is too short for some resources, for example RDS databases
}

import {
  to = aws_iam_role.spacelift
  id = local.iam_role_name
}
