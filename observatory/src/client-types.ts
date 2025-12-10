// Client-side types not yet in OpenAPI spec

export type TokenInfo = {
  id: string
  name: string
  created_at: string
  expiration_time: string
  last_used_at: string | null
}

export type TokenCreate = {
  name: string
}

export type TokenResponse = {
  token: string
}

export type TokenListResponse = {
  tokens: TokenInfo[]
}

export type TrainingRun = {
  id: string
  name: string
  created_at: string
  user_id: string
  finished_at: string | null
  status: string
  url: string | null
  description: string | null
  tags: string[]
}

export type TrainingRunListResponse = {
  training_runs: TrainingRun[]
}

export type TrainingRunPolicy = {
  policy_name: string
  policy_id: string
  epoch_start: number | null
  epoch_end: number | null
}

export type ValueOverReplacementSummary = {
  policy_version_id: string
  overall_vor: number | null
  overall_vor_std: number | null
  total_candidate_agents: number
}

// Client-side filter type (not from API)
export type TaskFilters = {
  command?: string
  user_id?: string
  status?: string
  assignee?: string
  git_hash?: string
  created_at?: string
  assigned_at?: string
}
