resource "aws_ssm_parameter" "skypilot_api_url" {
  name = "/skypilot/api_url"
  type = "String"

  # Note: [0] for each element are necessary, even though in yaml status looks like an object, not a list.
  # See https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/data-sources/service#example-usage
  value = "https://skypilot:${random_password.skypilot_password.result}@${var.subdomain}.${var.zone_domain}"
}

# route53 is configured by external-dns
removed {
  from = aws_route53_record.skypilot_api
}

removed {
  from = aws_route53_zone.main
}
