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

  # The data includes three fields:
  # - client-id
  # - client-secret
  # - cookie-secret
  # The cookie secret is useful as a value for OAUTH2_PROXY_COOKIE_SECRET.
  # Technically, each oauth-proxy deployment could have its own cookie secret, but it's easier to reuse it.
  data = jsondecode(data.aws_secretsmanager_secret_version.oauth_secret_version.secret_string)
}
