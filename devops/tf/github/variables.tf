// Generate here: https://github.com/settings/personal-access-tokens
// Needs Secrets rw scope.
variable "github_token" {
  type        = string
  description = "GitHub token"
  sensitive   = true
}
