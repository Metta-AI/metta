data "aws_secretsmanager_secret" "oauth_secret" {
  arn = var.oauth_secret_arn
}

data "aws_secretsmanager_secret_version" "oauth_secret_version" {
  secret_id = data.aws_secretsmanager_secret.oauth_secret.id
}

resource "kubernetes_secret" "oauth_secret" {
  for_each = toset(var.oauth_secret_namespaces)

  metadata {
    name      = "softmax-infra-oauth"
    namespace = each.value
  }

  data = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret_version.secret_string)
}
