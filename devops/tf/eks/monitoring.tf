removed {
  from = kubernetes_namespace.monitoring
  lifecycle {
    destroy = false
  }
}
