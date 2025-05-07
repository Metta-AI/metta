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
  task_role_arn = "arn:aws:iam::767406518141:instance-profile/skypilot-v1"

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
