# get load balancer from skypilot chart
data "kubernetes_service" "skypilot_ingress_nginx" {
  metadata {
    name = "skypilot-ingress-nginx-controller"
    namespace = "skypilot"
  }
  depends_on = [helm_release.skypilot]
}

resource "aws_ssm_parameter" "skypilot_api_url" {
  name  = "/skypilot/api_url"
  type  = "String"
  
  # Note: [0] for each element are necessary, even though in yaml status looks like an object, not a list.
  # See https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/data-sources/service#example-usage
  value = "https://skypilot:${random_password.skypilot_password.result}@${data.kubernetes_service.skypilot_ingress_nginx.status[0].load_balancer[0].ingress[0].hostname}"
}