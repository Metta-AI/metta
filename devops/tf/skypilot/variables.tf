variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "main"
}

variable "jobs_bucket" {
  default = "skypilot-jobs"
}

variable "db_identifier" {
  default = "skypilot-pg"
}

variable "db_instance_class" {
  type    = string
  default = "db.m8gd.large"
}

variable "db_allocated_storage" {
  type    = number
  default = 500 # GiB
}

variable "db_postgres_version" {
  type    = string
  default = "17.5"
}
