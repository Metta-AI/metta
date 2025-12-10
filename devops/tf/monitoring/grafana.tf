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
