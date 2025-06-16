resource "helm_release" "observatory" {
  name  = "observatory"
  chart = "./chart"

  namespace        = "observatory"
  create_namespace = true

  set {
    name  = "db_uri"
    value = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }
}
