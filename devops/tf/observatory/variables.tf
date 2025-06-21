variable "region" {
  type    = string
  default = "us-east-1"
}

variable "eks_cluster_name" {
  type    = string
  default = "main" # name from `eks` stack
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "db_allocated_storage" {
  type    = number
  default = 20 # GiB
}

variable "db_postgres_version" {
  type    = string
  default = "17.5"
}

variable "google_service_account_secret_arn" {
  type    = string
  default = "arn:aws:secretsmanager:us-east-1:751442549699:secret:GoogleOAuthObservatory-H7yGjS"
}
