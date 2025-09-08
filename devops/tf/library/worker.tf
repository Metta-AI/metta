resource "kubernetes_namespace" "library" {
  metadata {
    name = "library"
  }
}

resource "kubernetes_secret" "worker" {
  metadata {
    name      = var.worker_secret_name
    namespace = kubernetes_namespace.library.metadata[0].name
  }

  data = local.common_env_vars
}
