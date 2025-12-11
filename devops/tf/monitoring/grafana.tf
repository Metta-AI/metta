# TODO - move this to AWS secrets manager, and sync with external secrets.
# Then we can remove this stack (if we don't accumulate more stuff in it first).

resource "random_password" "grafana_password" {
  length  = 40
  special = false
}

resource "kubernetes_secret" "grafana" {
  metadata {
    # must be in sync with prometheus chart values
    name      = "grafana-credentials"
    namespace = "monitoring"
  }

  data = {
    admin-user     = "admin"
    admin-password = random_password.grafana_password.result
  }
}

# Grafana chart and prometheus are installed by helmfile.

resource "aws_iam_role" "grafana" {
  name = "grafana"

  assume_role_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "AllowEksAuthToAssumeRoleForPodIdentity",
        "Effect" : "Allow",
        "Principal" : {
          "Service" : "pods.eks.amazonaws.com"
        },
        "Action" : [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "grafana-cloudwatch" {
  role       = aws_iam_role.grafana.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"
}

resource "aws_eks_pod_identity_association" "grafana" {
  cluster_name    = data.aws_eks_cluster.main.name
  namespace       = kubernetes_namespace.monitoring.metadata[0].name
  service_account = "prometheus-grafana" # created by prometheus-kube-stack chart
  role_arn        = aws_iam_role.grafana.arn
}
