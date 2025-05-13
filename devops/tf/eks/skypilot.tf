module "skypilot" {
  source = "./skypilot"
}

moved {
    from = kubernetes_namespace.skypilot
    to = module.skypilot.kubernetes_namespace.skypilot
}

moved {
    from = helm_release.skypilot
    to = module.skypilot.helm_release.skypilot
}

moved {
    from = random_password.skypilot_password
    to = module.skypilot.random_password.skypilot_password
}

moved {
    from = aws_ssm_parameter.skypilot_api_url
    to = module.skypilot.aws_ssm_parameter.skypilot_api_url
}
