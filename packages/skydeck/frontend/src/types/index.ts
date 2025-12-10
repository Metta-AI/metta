export type JobStatus = 'INIT' | 'PENDING' | 'STARTING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED' | 'STOPPED' | 'UNKNOWN';
export type DesiredState = 'RUNNING' | 'STOPPED';

export interface Experiment {
  id: string;  // String representation of integer ID (auto-generated)
  name: string;
  base_command: string;
  tool_path: string | null;
  git_branch: string | null;
  nodes: number;
  gpus: number;
  instance_type: string | null;
  cloud: string | null;
  spot: boolean;
  flags: Record<string, unknown>;
  description: string | null;
  tags: string[];
  group: string | null;
  desired_state: DesiredState;
  current_state: string;
  current_job_id: string | null;
  starred: boolean;
  is_expanded: boolean;
  latest_epoch: number | null;
  exp_order: number;
}

export interface Job {
  id: string;
  experiment_id: string;
  status: JobStatus;
  nodes: number;
  gpus: number;
  command: string;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface Checkpoint {
  epoch: number;
  model_path: string | null;
  version: string | null;
  policy_version: string | null;
  policy_id: string | null;
  policy_version_id: string | null;
  observatory_url: string | null;
  created_at: string;
  replay_paths: string[];
}

export interface ExperimentGroup {
  id: string;
  name: string;
  name_prefix: string | null;
  flags: string[];  // Flag columns to display
  order: number;
  collapsed: boolean;
  experiments: Experiment[];  // Populated at runtime
}

export interface HealthBackend {
  staleness_seconds: number | null;
}

export interface HealthData {
  status: string;
  num_experiments: number;
  num_running_jobs: number;
  skypilot: HealthBackend | null;
  s3: HealthBackend | null;
  observatory: HealthBackend | null;
}

export interface NotificationData {
  id: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
  undoCallback?: () => void;
}

export interface OperationLog {
  id: number;
  timestamp: string;
  operation_type: 'START' | 'STOP' | 'CANCEL' | 'DELETE' | 'CREATE';
  experiment_id: number | null;
  experiment_name: string | null;
  job_id: string | null;
  success: boolean;
  error_message: string | null;
  output: string | null;
  user: string | null;
}
