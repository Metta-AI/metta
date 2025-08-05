resource "random_password" "grafana_password" {
  length  = 40
  special = false
}

# In the process of deploying everything from scratch, this terraform stack would run before we install charts.
# So we need to create the namespace first.
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
  }
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
