data "aws_caller_identity" "current" {}
locals {
  account_id = data.aws_caller_identity.current.account_id
}

resource "aws_iam_policy" "minimal" {
  name = "minimal-skypilot-policy"
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : "ec2:RunInstances",
        "Resource" : "arn:aws:ec2:*::image/ami-*"
      },
      {
        "Effect" : "Allow",
        "Action" : "ec2:RunInstances",
        "Resource" : [
          "arn:aws:ec2:*:${local.account_id}:instance/*",
          "arn:aws:ec2:*:${local.account_id}:network-interface/*",
          "arn:aws:ec2:*:${local.account_id}:subnet/*",
          "arn:aws:ec2:*:${local.account_id}:volume/*",
          "arn:aws:ec2:*:${local.account_id}:security-group/*"
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:TerminateInstances",
          "ec2:DeleteTags",
          "ec2:StartInstances",
          "ec2:CreateTags",
          "ec2:StopInstances",
        ],
        "Resource" : "arn:aws:ec2:*:${local.account_id}:instance/*"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:Describe*"
        ],
        "Resource" : "*"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:CreateSecurityGroup",
          "ec2:AuthorizeSecurityGroupIngress",
          "ec2:DeleteSecurityGroup"
        ],
        "Resource" : "arn:aws:ec2:*:${local.account_id}:*"
      },
      {
        "Effect" : "Allow",
        "Action" : "iam:CreateServiceLinkedRole",
        "Resource" : "*",
        "Condition" : {
          "StringEquals" : {
            "iam:AWSServiceName" : "spot.amazonaws.com"
          }
        }
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetRole",
          "iam:PassRole",
          # for skypilot-v1 role maintained by skypilot
          # disabled - we terraform the role
          # "iam:CreateRole",
          # "iam:AttachRolePolicy"
        ],
        "Resource" : [
          "arn:aws:iam::${local.account_id}:role/skypilot-v1"
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetInstanceProfile",
          # for skypilot-v1 role maintained by skypilot
          # we terraform the role but skypilot-api-server still needs these permissions
          "iam:CreateInstanceProfile",
          "iam:AddRoleToInstanceProfile"
        ],
        "Resource" : "arn:aws:iam::${local.account_id}:instance-profile/skypilot-v1"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:*"
        ],
        "Resource" : "*"
      },
      # ECR read-only (identical to AmazonEC2ContainerRegistryReadOnly)
      {
        "Effect" : "Allow",
        "Action" : [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:GetLifecyclePolicy",
          "ecr:GetLifecyclePolicyPreview",
          "ecr:ListTagsForResource",
          "ecr:DescribeImageScanFindings"
        ],
        "Resource" : "*"
      },
    ]
  })
}

# User - for API server that runs on EKS
resource "aws_iam_user" "skypilot_api_server" {
  name = "skypilot-api-server"
}

resource "aws_iam_user_policy_attachment" "skypilot_api_server_attach" {
  user       = aws_iam_user.skypilot_api_server.name
  policy_arn = aws_iam_policy.minimal.arn
}

# Role - for EC2 instances that are launched by skypilot API server
resource "aws_iam_role" "skypilot_v1" {
  name = "skypilot-v1"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "skypilot_v1_attach" {
  role       = aws_iam_role.skypilot_v1.name
  policy_arn = aws_iam_policy.minimal.arn
}

resource "aws_iam_access_key" "skypilot_api_server" {
  user = aws_iam_user.skypilot_api_server.name
}
