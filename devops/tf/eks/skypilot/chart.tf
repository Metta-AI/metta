resource "random_password" "skypilot_password" {
  length = 40
  special = false
}

resource "helm_release" "skypilot" {
  name       = "skypilot"

  # Using our local fork, see ./README.md for details
  # repository = "https://helm.skypilot.co"
  # chart      = "skypilot-nightly"

  # relative to stack root
  chart      = "./skypilot/skypilot-chart"
  dependency_update = true

  devel      = true
  namespace  = "skypilot"
  create_namespace = true

  set_sensitive {
    name  = "ingress.authCredentials"
    value = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }

  set {
    name  = "ingress-nginx.controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"
    value = "internet-facing"
  }

  set {
    name = "ingress.certManager.enabled"
    value = "true"
  }

  set {
    name = "ingress.certManager.clusterIssuer"
    value = "letsencrypt"
  }

  set {
    name = "ingress.host"
    value = "${var.subdomain}.${var.zone_domain}"
  }

  set {
    name = "awsCredentials.enabled"
    value = "true"
  }
}

resource "kubernetes_secret" "skypilot_api_server_credentials" {
  metadata {
    name = "aws-credentials" # default name in skypilot chart
    namespace = "skypilot"
  }

  data = {
    aws_access_key_id = aws_iam_access_key.skypilot_api_server.id
    aws_secret_access_key = aws_iam_access_key.skypilot_api_server.secret
  }
}
