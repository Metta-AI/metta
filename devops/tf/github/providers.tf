terraform {
  required_providers {
    github = { source = "integrations/github" }
  }
}

provider "github" {
  token = var.github_token
  owner = "Metta-AI"
}
