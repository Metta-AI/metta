resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

resource "kubernetes_secret" "observatory" {
  metadata {
    name      = "observatory-env"
    namespace = "observatory"
  }

  data = {
    STATS_DB_URI = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }
}
