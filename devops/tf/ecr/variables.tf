variable "region" {
  type    = string
  default = "us-east-1"
}

variable "replication_regions" {
  type    = list(string)
  default = ["us-west-2"]
}
