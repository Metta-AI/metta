# Imports for migration from the previous module in eks stack
import {
  to = aws_db_instance.postgres
  id = var.db_identifier
}

import {
  to = aws_iam_policy.minimal
  id = "arn:aws:iam::751442549699:policy/minimal-skypilot-policy"
}

import {
  to = aws_iam_user.skypilot_api_server
  id = "skypilot-api-server"
}

import {
  to = aws_iam_user_policy_attachment.skypilot_api_server_attach
  id = "skypilot-api-server/arn:aws:iam::751442549699:policy/minimal-skypilot-policy"
}

import {
  to = aws_iam_role.skypilot_v1
  id = "skypilot-v1"
}

import {
  to = aws_iam_role_policy_attachment.skypilot_v1_attach
  id = "skypilot-v1/arn:aws:iam::751442549699:policy/minimal-skypilot-policy"
}

import {
  to = aws_iam_access_key.skypilot_api_server
  id = "AKIA255LXLPB3MU6GBML"
}

import {
  to = aws_s3_bucket.skypilot_jobs
  id = var.jobs_bucket
}
