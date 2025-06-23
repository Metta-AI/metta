terraform {
  required_providers {
    spacelift = { source = "spacelift-io/spacelift" }
  }
}

provider "aws" {
  region = var.aws_zone
}
