resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

resource "kubernetes_secret" "observatory_backend_env" {
  metadata {
    name      = "observatory-backend-env"
    namespace = kubernetes_namespace.observatory.metadata[0].name
  }
  data = {
    STATS_DB_URI = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }
}
