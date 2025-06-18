resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

resource "helm_release" "observatory" {
  name  = "observatory"
  chart = "./chart"

  namespace = kubernetes_namespace.observatory.metadata[0].name


  force_update = true

  set {
    name  = "db_uri"
    value = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }

  set {
    name  = "host"
    value = var.api_host
  }
}
