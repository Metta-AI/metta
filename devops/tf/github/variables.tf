variable "aws_zone" {
  type    = string
  default = "us-east-1"
}

// Generate here: https://github.com/settings/personal-access-tokens
// Needs Secrets rw scope.
variable "github_token" {
  type        = string
  description = "GitHub token"
  sensitive   = true
}

variable "github_org" {
  type    = string
  default = "Metta-AI"
}

variable "github_repo" {
  type    = string
  default = "metta"
}

variable "extra_oidc_repos" {
  # Allow github actions to access AWS from these repos.
  type    = list(string)
  default = ["alignsoftmax/softmaxdotcom"]
}
