module "skypilot" {
  source = "./skypilot"

  depends_on = [aws_route53_zone.softmax]
}
