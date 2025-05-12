
resource "kubernetes_namespace" "skypilot" {
  metadata {
    name = "skypilot"
  }
}

resource "random_password" "skypilot_password" {
  length = 40
}

resource "helm_release" "skypilot" {
  name       = "skypilot"
  repository = "https://helm.skypilot.co"
  chart      = "skypilot-nightly"
  devel = true
  namespace  = kubernetes_namespace.skypilot.metadata[0].name

  set {
    name  = "ingress.authCredentials"
    value = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }
}
