resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true

  set = [
    {
      name  = "installCRDs"
      value = "true"
    }
  ]
}

# It's not possible to install the clusterissuer as kubernetes_manifest resource,
# because it relies on CRDs that aren't installed yet; the resource would fail at the
# plan-time.
#
# So we install the clusterissuer via helm.
#
# See also: https://github.com/hashicorp/terraform-provider-kubernetes/issues/1367#issuecomment-2277333258
resource "helm_release" "clusterissuer" {
  name       = "clusterissuer"
  chart      = "./cert-manager-configs"
  namespace  = "cert-manager"
  depends_on = [helm_release.cert_manager]
}
