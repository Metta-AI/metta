resource "aws_eks_pod_identity_association" "replay_mining" {
  cluster_name    = module.eks.cluster_name
  namespace       = "monitoring"
  service_account = "replay-mining-cronjob"
  role_arn        = aws_iam_role.replay_mining.arn
}

resource "aws_iam_role" "replay_mining" {
  name = "replay-mining-cronjob"
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
}

resource "aws_iam_role_policy_attachment" "replay_mining_s3" {
  role       = aws_iam_role.replay_mining.name
  policy_arn = aws_iam_policy.replay_mining_s3.arn
}

resource "aws_iam_policy" "replay_mining_s3" {
  name = "replay-mining-s3-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
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

output "replay_mining_role_arn" {
  value       = aws_iam_role.replay_mining.arn
  description = "ARN of the IAM role for replay mining cronjob service account"
}
