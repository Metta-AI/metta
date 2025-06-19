# get load balancer from observatory chart
data "kubernetes_ingress" "observatory" {
  metadata {
    name      = "observatory"
    namespace = "observatory"
  }
  depends_on = [helm_release.observatory]
}

locals {
  lb_hostname = data.kubernetes_ingress.observatory.status[0].load_balancer[0].ingress[0].hostname
}

data "aws_route53_zone" "main" {
  name = var.zone_domain
}

resource "aws_route53_record" "observatory_api" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.api_host
  type    = "CNAME"
  ttl     = 60

  records = [local.lb_hostname]
}
