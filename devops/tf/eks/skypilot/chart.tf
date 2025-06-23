# moved to devops/charts/, we don't want to manage helm charts with terraform
removed {
  from = helm_release.skypilot
}
