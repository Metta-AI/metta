locals {
  skypilot_api_port  = 46580
  skypilot_api_image = "berkeleyskypilot/skypilot-nightly:latest"
  skypilot_api_cpu   = "8192"  # 8 vCPU
  skypilot_api_mem   = "16384" # 16GB
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
  memory                   = local.skypilot_api_mem
  execution_role_arn       = aws_iam_role.ecs_task_exec.arn

  container_definitions = jsonencode([
    {
      name      = "skypilot"
      image     = local.skypilot_api_image
      essential = true
      portMappings = [{
        containerPort = local.skypilot_api_port
        protocol      = "tcp"
      }]
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "skypilot_service" {
  name            = "skypilot-api-server"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.skypilot_api_server.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_default_subnet.proxy_subnet.id]
    security_groups = [aws_security_group.alb_sg.id]
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
