resource "helm_release" "observatory" {
  name  = "observatory"
  chart = "./chart"

  namespace        = "observatory"
  create_namespace = true
}
