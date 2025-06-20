terraform {
  required_providers {
    github = { source = "integrations/github" }
  }
}

provider "aws" {
  region = var.aws_zone
}

provider "github" {
  token = var.github_token
  owner = var.github_org
}
