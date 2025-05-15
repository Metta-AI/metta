resource "kubernetes_storage_class" "gp3_default" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "true"
    }
  }

  # correct provisioner for EKS auto mode, https://stackoverflow.com/a/79601460
  storage_provisioner = "ebs.csi.eks.amazonaws.com"

  volume_binding_mode = "WaitForFirstConsumer"
  parameters = {
    type      = "gp3"
    encrypted = "true"
  }
}
