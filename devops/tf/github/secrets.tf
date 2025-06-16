data "aws_ssm_parameter" "skypilot_api_url" {
  name = "/skypilot/api_url"
}

resource "github_actions_secret" "skypilot_api_url" {
  repository      = "metta"
  secret_name     = "SKYPILOT_API_URL"
  plaintext_value = data.aws_ssm_parameter.skypilot_api_url.value
}
