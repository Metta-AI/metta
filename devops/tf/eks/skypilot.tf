
resource "kubernetes_namespace" "skypilot" {
  metadata {
    name = "skypilot"
  }
}

resource "random_password" "skypilot_password" {
  length = 40
}

# Deployment for this chart is patched manually in production with `--host 0.0.0.0`
# See also: https://skypilot-org.slack.com/archives/C03J2KQQZSS/p1747063075515989?thread_ts=1746545299.574249&cid=C03J2KQQZSS
# (I triggered the same bug and used the same fix.)
resource "helm_release" "skypilot" {
  name       = "skypilot"
  repository = "https://helm.skypilot.co"
  chart      = "skypilot-nightly"
  devel = true
  namespace  = kubernetes_namespace.skypilot.metadata[0].name

  set_sensitive {
    name  = "ingress.authCredentials"
    value = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }

  set {
    name  = "ingress-nginx.controller.service.annotations"
    value = "service.beta.kubernetes.io/aws-load-balancer-scheme: \"internet-facing\""
  }
}
