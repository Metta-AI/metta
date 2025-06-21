resource "random_password" "skypilot_password" {
  length  = 40
  special = false
}

resource "kubernetes_secret" "skypilot_auth" {
  metadata {
    name      = "skypilot-basic-auth" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    auth = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }
}

resource "helm_release" "skypilot" {
  name = "skypilot"

  # Using our local fork, see ./README.md for details
  # repository = "https://helm.skypilot.co"
  # chart      = "skypilot-nightly"

  # relative to stack root
  chart             = "./skypilot/skypilot-chart"
  dependency_update = true

  devel            = true
  namespace        = "skypilot"
  create_namespace = true

  set = [
    {
      name  = "ingress-nginx.controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"
      value = "internet-facing"
    },

    {
      name  = "ingress.certManager.enabled"
      value = "true"
    },

    {
      name  = "ingress.certManager.clusterIssuer"
      value = "letsencrypt"
    },

    {
      name  = "ingress.host"
      value = "${var.subdomain}.${var.zone_domain}"
    },

    {
      name  = "awsCredentials.enabled"
      value = "true"
    },

    {
      name  = "lambdaAiCredentials.enabled"
      value = "true"
    },

    {
      name = "apiService.config"
      value = templatefile("${path.module}/skypilot-config.tftpl", {
        jobs_bucket = aws_s3_bucket.skypilot_jobs.bucket
      })
    },

    {
      name  = "lambdaAiCredentials.lambdaAiSecretName"
      value = kubernetes_secret.lambda_ai_secret.metadata[0].name
    },

    {
      name  = "ingress.authSecret"
      value = kubernetes_secret.skypilot_auth.metadata[0].name
    }
  ]
}

resource "kubernetes_secret" "skypilot_api_server_credentials" {
  metadata {
    name      = "aws-credentials" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    aws_access_key_id     = aws_iam_access_key.skypilot_api_server.id
    aws_secret_access_key = aws_iam_access_key.skypilot_api_server.secret
  }
}
