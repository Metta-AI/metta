removed {
  from = kubernetes_namespace.monitoring
  # requries opentofu 1.11
  lifecycle {
    destroy = false
  }
}
