# switched to `create_namespace = true`
removed {
  from = kubernetes_namespace.skypilot
}

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
}
