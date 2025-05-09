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
          # disabled - we terraform the role
          # "iam:CreateRole",
          # "iam:AttachRolePolicy"
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
          # disabled - we terraform the role
          # "iam:CreateInstanceProfile",
          # "iam:AddRoleToInstanceProfile"
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

resource "aws_iam_role" "skypilot_v1" {
  name = "skypilot-v1"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
      }
    ]
  })
}


resource "aws_iam_role_policy_attachment" "skypilot_v1_attach" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/AmazonEC2FullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
  ])
  role       = aws_iam_role.skypilot_v1.name
  policy_arn = each.value
}

resource "aws_iam_role_policy_attachment" "skypilot_api_server_ssm" {
  role       = aws_iam_role.skypilot_api_server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}


# For some reason, skypilot didn't initialize skypilot-v1 role correctly.
# Old account had this inline policy, but the new account only had EC2 and S3 attached.
# So I'm reproducing the inline policy here; I think this is necessary for managed jobs.
resource "aws_iam_role_policy" "skypilot_v1_pass_role" {
  name = "SkyPilotPassRolePolicy"
  role = aws_iam_role.skypilot_v1.name
  policy = jsonencode({
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : [
          "iam:GetRole",
          "iam:PassRole"
        ],
        "Resource" : "arn:aws:iam::${local.account_id}:role/skypilot-v1"
      },
      {
        "Effect" : "Allow",
        "Action" : "iam:GetInstanceProfile",
        "Resource" : "arn:aws:iam::${local.account_id}:instance-profile/skypilot-v1"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "skypilot_v1" {
  name = "skypilot-v1"
  role = aws_iam_role.skypilot_v1.name
}

# ECS Cluster
resource "aws_ecs_cluster" "skypilot_api_server" {
  name = "skypilot-api-server"
}

# Data
resource "aws_efs_file_system" "skypilot_api_server_data" {
  creation_token = "skypilot-api-server-data"
  tags = {
    Name = "skypilot-api-server-data"
  }
}

resource "aws_efs_mount_target" "skypilot_api_server_data" {
  file_system_id  = aws_efs_file_system.skypilot_api_server_data.id
  subnet_id       = aws_default_subnet.proxy_subnet.id
  security_groups = [aws_security_group.allow_efs_access.id]
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

  volume {
    name = "skypilot-root-sky-volume"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.skypilot_api_server_data.id
      transit_encryption = "ENABLED"
    }
  }

  container_definitions = jsonencode([
    {
      name = "skypilot"
      command = ["/bin/sh", "-c", <<-EOT
        # persist data
        ln -s /mnt/data /root/.sky

        # except for tmp which skypilot uses for its own mounts
        mkdir -p /tmp/sky-tmp
        rm -rf /root/.sky/tmp
        ln -s /tmp/sky-tmp /root/.sky/tmp

        # start the api server
        sky api start --deploy --foreground
      EOT
      ]
      image     = local.skypilot_api_image
      essential = true
      portMappings = [{
        containerPort = local.skypilot_api_port
      }]
      mountPoints = [
        {
          sourceVolume  = "skypilot-root-sky-volume"
          containerPath = "/mnt/data"
          readOnly      = false
        }
      ]
      healthCheck = {
        command     = ["CMD", "curl", "-f", "http://localhost:${local.skypilot_api_port}/api/health"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
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

  enable_execute_command = true

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
