
resource "kubernetes_namespace" "skypilot" {
  metadata {
    name = "skypilot"
  }
}

resource "random_password" "skypilot_password" {
  length = 40
}

# Deployment for this chart is patched manually in production with `--host 0.0.0.0`
# See also: https://skypilot-org.slack.com/archives/C03J2KQQZSS/p1747063075515989?thread_ts=1746545299.574249&cid=C03J2KQQZSS
# (I triggered the same bug and used the same fix.)
#
# If you're modifying it and terraform wants to update or recreate, you might need to apply the patch with `kubectl edit` again.
resource "helm_release" "skypilot" {
  name       = "skypilot"
  repository = "https://helm.skypilot.co"
  chart      = "skypilot-nightly"
  devel      = true
  namespace  = kubernetes_namespace.skypilot.metadata[0].name

  set_sensitive {
    name  = "ingress.authCredentials"
    value = "skypilot:${bcrypt(random_password.skypilot_password.result)}"
  }

  set {
    name  = "ingress-nginx.controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"
    value = "internet-facing"
  }
}

# get load balancer from skypilot chart
data "kubernetes_service" "skypilot_ingress_nginx" {
  metadata {
    name = "skypilot-ingress-nginx-controller"
    namespace = "skypilot"
  }
}

resource "aws_ssm_parameter" "skypilot_api_url" {
  name  = "/skypilot/api_url"
  type  = "String"
  
  # Note: [0] for each element are necessary, even though in yaml status looks like an object, not a list.
  # See https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/data-sources/service#example-usage
  value = "https://skypilot:${random_password.skypilot_password.result}@${data.kubernetes_service.skypilot_ingress_nginx.status[0].load_balancer[0].ingress[0].hostname}"
}
