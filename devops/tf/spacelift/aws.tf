data "aws_caller_identity" "current" {}

locals {
  aws_integration_name = "softmax-aws"
  iam_role_name        = "Spacelift"
  iam_role_arn         = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${local.iam_role_name}"
}

# Create IAM role for Spacelift to assume
resource "aws_iam_role" "spacelift" {
  name = local.iam_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
        ... # TODO
    ]
  })
}

# Make the role a poweruser
resource "aws_iam_role_policy_attachment" "spacelift_poweruser" {
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
  role       = aws_iam_role.spacelift.name
}


# Register the role with Spacelift
resource "spacelift_aws_integration" "softmax" {
  name = local.aws_integration_name

  role_arn = local.iam_role_arn
  depends_on = [
    aws_iam_role.spacelift
  ]
}

