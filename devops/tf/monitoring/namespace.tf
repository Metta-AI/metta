import {
  to = kubernetes_namespace.monitoring
  id = "monitoring"
}

# In the process of deploying everything from scratch, this terraform stack would run before we install charts.
# So we need to create the namespace first.
resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
  }
}
