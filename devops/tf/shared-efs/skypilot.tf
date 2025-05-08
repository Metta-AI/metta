locals {
  skypilot_api_image  = "berkeleyskypilot/skypilot-nightly:latest"
  skypilot_api_cpu    = "8192"  # 8 vCPU
  skypilot_api_memory = "16384" # 16GB
}

resource "aws_security_group" "allow_skypilot_inbound" {
  name        = "skypilot-inbound"
  description = "Allow HTTP inbound"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
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
          "arn:aws:ec2:*:<account-ID-without-hyphens>:instance/*",
          "arn:aws:ec2:*:<account-ID-without-hyphens>:network-interface/*",
          "arn:aws:ec2:*:<account-ID-without-hyphens>:subnet/*",
          "arn:aws:ec2:*:<account-ID-without-hyphens>:volume/*",
          "arn:aws:ec2:*:<account-ID-without-hyphens>:security-group/*"
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
        "Resource" : "arn:aws:ec2:*:<account-ID-without-hyphens>:instance/*"
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
        "Resource" : "arn:aws:ec2:*:<account-ID-without-hyphens>:*"
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
          # for skypilot-v1 role
          "iam:CreateRole",
          "iam:AttachRolePolicy"
        ],
        "Resource" : [
          "arn:aws:iam::<account-ID-without-hyphens>:role/skypilot-v1"
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetInstanceProfile",
          # for skypilot-v1 role
          "iam:CreateInstanceProfile",
          "iam:AddRoleToInstanceProfile"
        ],
        "Resource" : "arn:aws:iam::<account-ID-without-hyphens>:instance-profile/skypilot-v1"
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
        containerPort = 46580
        hostPort      = 80
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
    subnets         = [aws_default_subnet.proxy_subnet.id]
    security_groups = [aws_security_group.allow_skypilot_inbound.id]
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
