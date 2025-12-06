terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
    datadog = {
      source = "DataDog/datadog"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

data "aws_secretsmanager_secret" "datadog_api_key" {
  name = "datadog/api-key"
}

data "aws_secretsmanager_secret_version" "datadog_api_key_version" {
  secret_id = data.aws_secretsmanager_secret.datadog_api_key.id
}

data "aws_secretsmanager_secret" "datadog_app_key" {
  name = "datadog/app-key"
}

data "aws_secretsmanager_secret_version" "datadog_app_key_version" {
  secret_id = data.aws_secretsmanager_secret.datadog_app_key.id
}

provider "datadog" {
  api_key = data.aws_secretsmanager_secret_version.datadog_api_key_version.secret_string
  app_key = data.aws_secretsmanager_secret_version.datadog_app_key_version.secret_string
}
