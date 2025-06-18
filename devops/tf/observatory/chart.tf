resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

# `observatory-secrets` secret was created manually
# (it's not possible to automate it because it's not possible to terraform oauth clients in GCP)

resource "helm_release" "observatory" {
  name  = "observatory"
  chart = "./chart"

  namespace = kubernetes_namespace.observatory.metadata[0].name

  set {
    name  = "db_uri"
    value = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }
}
