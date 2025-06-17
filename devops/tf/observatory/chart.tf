data "aws_secretsmanager_secret" "google_service_account_secret" {
  arn = var.google_service_account_secret_arn
}

data "aws_secretsmanager_secret_version" "google_service_account_secret_version" {
  secret_id = data.aws_secretsmanager_secret.google_service_account_secret
}

resource "kubernetes_namespace" "observatory" {
  metadata {
    name = "observatory"
  }
}

resource "kubernetes_secret" "observatory_secrets" {
  metadata {
    name      = "observatory-secrets"
    namespace = kubernetes_namespace.observatory.metadata[0].name
  }

  data = {
    "google-service-account.json" = data.aws_secretsmanager_secret_version.google_service_account_secret_version.secret_string
  }
}

resource "helm_release" "observatory" {
  name  = "observatory"
  chart = "./chart"

  namespace = kubernetes_namespace.observatory.metadata[0].name

  set {
    name  = "db_uri"
    value = "postgresql://${aws_db_instance.postgres.username}:${aws_db_instance.postgres.password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  }
}
