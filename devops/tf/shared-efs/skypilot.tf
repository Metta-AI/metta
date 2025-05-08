locals {
  skypilot_api_port   = 46580
  skypilot_api_image  = "berkeleyskypilot/skypilot-nightly:latest"
  skypilot_api_cpu    = "8192"  # 8 vCPU
  skypilot_api_memory = "16384" # 16GB
  account_id          = data.aws_caller_identity.current.account_id
}

data "aws_caller_identity" "current" {}

resource "aws_security_group" "allow_skypilot_inbound" {
  name        = "skypilot-inbound"
  description = "Allow inbound traffic to skypilot API server"
  vpc_id      = var.vpc_id

  # allow access from proxy
  ingress {
    from_port       = local.skypilot_api_port
    to_port         = local.skypilot_api_port
    protocol        = "tcp"
    security_groups = [aws_security_group.proxy.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# IAM
resource "aws_iam_role" "skypilot_api_server" {
  name               = "skypilot-api-server"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy" "skypilot_api_server" {
  name = "minimal-skypilot-policy"
  role = aws_iam_role.skypilot_api_server.id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : "ec2:RunInstances",
        "Resource" : "arn:aws:ec2:*::image/ami-*"
      },
      {
        "Effect" : "Allow",
        "Action" : "ec2:RunInstances",
        "Resource" : [
          "arn:aws:ec2:*:${local.account_id}:instance/*",
          "arn:aws:ec2:*:${local.account_id}:network-interface/*",
          "arn:aws:ec2:*:${local.account_id}:subnet/*",
          "arn:aws:ec2:*:${local.account_id}:volume/*",
          "arn:aws:ec2:*:${local.account_id}:security-group/*"
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:TerminateInstances",
          "ec2:DeleteTags",
          "ec2:StartInstances",
          "ec2:CreateTags",
          "ec2:StopInstances"
        ],
        "Resource" : "arn:aws:ec2:*:${local.account_id}:instance/*"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:Describe*"
        ],
        "Resource" : "*"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "ec2:CreateSecurityGroup",
          "ec2:AuthorizeSecurityGroupIngress"
        ],
        "Resource" : "arn:aws:ec2:*:${local.account_id}:*"
      },
      {
        "Effect" : "Allow",
        "Action" : "iam:CreateServiceLinkedRole",
        "Resource" : "*",
        "Condition" : {
          "StringEquals" : {
            "iam:AWSServiceName" : "spot.amazonaws.com"
          }
        }
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetRole",
          "iam:PassRole",
          # for skypilot-v1 role maintained by skypilot
          "iam:CreateRole",
          "iam:AttachRolePolicy"
        ],
        "Resource" : [
          "arn:aws:iam::${local.account_id}:role/skypilot-v1"
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetInstanceProfile",
          # for skypilot-v1 role maintained by skypilot
          "iam:CreateInstanceProfile",
          "iam:AddRoleToInstanceProfile"
        ],
        "Resource" : "arn:aws:iam::${local.account_id}:instance-profile/skypilot-v1"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:*"
        ],
        "Resource" : "*"
      }
    ]
  })
}


# ECS Cluster
resource "aws_ecs_cluster" "skypilot_api_server" {
  name = "skypilot-api-server"
}

# ECS Task Definition (Fargate)
resource "aws_ecs_task_definition" "skypilot_api_server" {
  family                   = "skypilot-api-server-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = local.skypilot_api_cpu
  memory                   = local.skypilot_api_memory

  execution_role_arn = aws_iam_role.ecs_task_exec.arn
  # give AWS credentials to the api server
  task_role_arn = aws_iam_role.skypilot_api_server.arn

  container_definitions = jsonencode([
    {
      name      = "skypilot"
      command   = ["sky", "api", "start", "--deploy", "--foreground"]
      image     = local.skypilot_api_image
      essential = true
      portMappings = [{
        containerPort = local.skypilot_api_port
      }]
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "skypilot_service" {
  name            = "skypilot-api-server"
  cluster         = aws_ecs_cluster.skypilot_api_server.id
  task_definition = aws_ecs_task_definition.skypilot_api_server.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_default_subnet.proxy_subnet.id]
    security_groups  = [aws_security_group.allow_skypilot_inbound.id]
    assign_public_ip = true # required to pull docker image
  }

  service_registries {
    registry_arn = aws_service_discovery_service.skypilot_api_server.arn
  }
}

resource "aws_service_discovery_service" "skypilot_api_server" {
  name = "skypilot-api-server"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.dns.id
    dns_records {
      type = "A"
      ttl  = 30
    }
  }
}
