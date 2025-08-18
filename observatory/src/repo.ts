export type ScorecardCell = {
  evalName: string
  replayUrl: string | null
  thumbnailUrl: string | null
  value: number
}

export type ScorecardData = {
  evalNames: string[]
  cells: Record<string, Record<string, ScorecardCell>>
  policyAverageScores: Record<string, number>
  evalAverageScores: Record<string, number>
  evalMaxScores: Record<string, number>
}

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

// Dashboard state interface for saving/loading
export interface DashboardState {
  selectedTrainingRunIds: string[]
  selectedRunFreePolicyIds: string[]
  selectedEvalNames: string[]
  trainingRunPolicySelector: 'latest' | 'best'
  selectedMetric: string
}

export type SavedDashboard = {
  id: string
  name: string
  description: string | null
  type: string
  dashboard_state: DashboardState
  created_at: string
  updated_at: string
  user_id: string
}

export type SavedDashboardCreate = {
  name: string
  description?: string
  type: string
  dashboard_state: Record<string, any>
}

export type SavedDashboardListResponse = {
  dashboards: SavedDashboard[]
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

export type TrainingRunDescriptionUpdate = {
  description: string
}

export type TrainingRunTagsUpdate = {
  tags: string[]
}

export type EvalTaskCreateRequest = {
  policy_id: string
  git_hash: string | null
  env_overrides?: Record<string, any>
  sim_suite?: string
}

export type EvalTask = {
  id: string
  policy_id: string
  sim_suite: string
  status: 'unprocessed' | 'canceled' | 'done' | 'error'
  assigned_at: string | null
  assignee: string | null
  created_at: string
  attributes: Record<string, any>
  policy_name: string | null
  retries: number
  updated_at: string
  user_id: string | null
}

export type EvalTasksResponse = {
  tasks: EvalTask[]
}

// Policy-based scorecard types
export type PaginationRequest = {
  page: number
  page_size: number
}

export type TrainingRunInfo = {
  id: string
  name: string
  user_id: string | null
  created_at: string
  tags: string[]
}

export type RunFreePolicyInfo = {
  id: string
  name: string
  user_id: string | null
  created_at: string
}

export type UnifiedPolicyInfo = {
  id: string
  type: 'training_run' | 'policy'
  name: string
  user_id: string | null
  created_at: string
  tags: string[]
}

export type PoliciesResponse = {
  policies: UnifiedPolicyInfo[]
}

export type EvalNamesRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
}

export type MetricsRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
  eval_names: string[]
}

export type PolicyScorecardRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
  eval_names: string[]
  training_run_policy_selector: 'latest' | 'best'
  metric: string
}

export type TrainingRunScorecardRequest = {
  eval_names: string[]
  metric: string
}

export type TrainingRunPolicy = {
  policy_name: string
  policy_id: string
  epoch_start: number | null
  epoch_end: number | null
}

export type PolicyScorecardCell = {
  evalName: string
  replayUrl: string | null
  thumbnailUrl: string | null
  value: number
}

export type PolicyScorecardData = {
  evalNames: string[]
  policyNames: string[]
  cells: Record<string, Record<string, PolicyScorecardCell>>
  policyAverageScores: Record<string, number>
  evalAverageScores: Record<string, number>
  evalMaxScores: Record<string, number>
}

// Leaderboard types
export type Leaderboard = {
  id: string
  name: string
  user_id: string
  evals: string[]
  metric: string
  start_date: string
  latest_episode: number
  created_at: string
  updated_at: string
}

export type LeaderboardCreateOrUpdate = {
  name: string
  evals: string[]
  metric: string
  start_date: string
}

export type LeaderboardListResponse = {
  leaderboards: Leaderboard[]
}

export type LeaderboardScorecardRequest = {
  selector: 'latest' | 'best'
  num_policies: number
}

import { config } from './config'

export type TableInfo = {
  table_name: string
  column_count: number
  row_count: number
}

export type TableSchema = {
  table_name: string
  columns: Array<{
    name: string
    type: string
    nullable: boolean
    default: string | null
    max_length: number | null
  }>
}

export type SQLQueryRequest = {
  query: string
}

export type SQLQueryResponse = {
  columns: string[]
  rows: any[][]
  row_count: number
}

export type AIQueryRequest = {
  description: string
}

export type AIQueryResponse = {
  query: string
}

/**
 * Interface for data fetching.
 *
 * Currently the data is loaded from a pre-computed JSON file.
 * In the future, we will fetch the data from an API.
 */
export interface Repo {
  // Token management methods
  createToken(tokenData: TokenCreate): Promise<TokenResponse>
  listTokens(): Promise<TokenListResponse>
  deleteToken(tokenId: string): Promise<void>

  // Saved dashboard methods
  listSavedDashboards(): Promise<SavedDashboardListResponse>
  getSavedDashboard(dashboardId: string): Promise<SavedDashboard>
  createSavedDashboard(dashboardData: SavedDashboardCreate): Promise<SavedDashboard>
  updateDashboardState(dashboardId: string, dashboardState: DashboardState): Promise<SavedDashboard>
  deleteSavedDashboard(dashboardId: string): Promise<void>

  // User methods
  whoami(): Promise<{ user_email: string }>

  // SQL query methods
  listTables(): Promise<TableInfo[]>
  getTableSchema(tableName: string): Promise<TableSchema>
  executeQuery(request: SQLQueryRequest): Promise<SQLQueryResponse>
  generateAIQuery(description: string): Promise<AIQueryResponse>

  // Training run methods
  getTrainingRuns(): Promise<TrainingRunListResponse>
  getTrainingRun(runId: string): Promise<TrainingRun>
  updateTrainingRunDescription(runId: string, description: string): Promise<TrainingRun>
  updateTrainingRunTags(runId: string, tags: string[]): Promise<TrainingRun>
  generateTrainingRunScorecard(runId: string, request: TrainingRunScorecardRequest): Promise<PolicyScorecardData>
  getTrainingRunPolicies(runId: string): Promise<TrainingRunPolicy[]>

  // Eval task methods
  createEvalTask(request: EvalTaskCreateRequest): Promise<EvalTask>
  getEvalTasks(): Promise<EvalTask[]>

  // Policy methods
  getPolicyIds(policyNames: string[]): Promise<Record<string, string>>

  // Policy-based scorecard methods
  getPolicies(): Promise<PoliciesResponse>
  getEvalNames(request: EvalNamesRequest): Promise<Set<string>>
  getAvailableMetrics(request: MetricsRequest): Promise<string[]>
  generatePolicyScorecard(request: PolicyScorecardRequest): Promise<PolicyScorecardData>

  // Leaderboard methods
  listLeaderboards(): Promise<LeaderboardListResponse>
  getLeaderboard(leaderboardId: string): Promise<Leaderboard>
  createLeaderboard(leaderboardData: LeaderboardCreateOrUpdate): Promise<Leaderboard>
  updateLeaderboard(leaderboardId: string, leaderboardData: LeaderboardCreateOrUpdate): Promise<Leaderboard>
  deleteLeaderboard(leaderboardId: string): Promise<void>
  generateLeaderboardScorecard(leaderboardId: string, request: LeaderboardScorecardRequest): Promise<ScorecardData>
}

export class ServerRepo implements Repo {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  private getHeaders(contentType?: string): Record<string, string> {
    const headers: Record<string, string> = {}

    if (contentType) {
      headers['Content-Type'] = contentType
    }

    if (config.authToken) {
      headers['X-Auth-Token'] = config.authToken
    }

    return headers
  }

  private async apiCall<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: this.getHeaders(),
    })
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallWithBody<T>(endpoint: string, body: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.getHeaders('application/json'),
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallWithBodyPut<T>(endpoint: string, body: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'PUT',
      headers: this.getHeaders('application/json'),
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallDelete(endpoint: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    })
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
  }

  // Token management methods
  async createToken(tokenData: TokenCreate): Promise<TokenResponse> {
    return this.apiCallWithBody<TokenResponse>('/tokens', tokenData)
  }

  async listTokens(): Promise<TokenListResponse> {
    return this.apiCall<TokenListResponse>('/tokens')
  }

  async deleteToken(tokenId: string): Promise<void> {
    return this.apiCallDelete(`/tokens/${tokenId}`)
  }

  // Saved dashboard methods
  async listSavedDashboards(): Promise<SavedDashboardListResponse> {
    return this.apiCall<SavedDashboardListResponse>('/dashboard/saved')
  }

  async getSavedDashboard(dashboardId: string): Promise<SavedDashboard> {
    return this.apiCall<SavedDashboard>(`/dashboard/saved/${encodeURIComponent(dashboardId)}`)
  }

  async createSavedDashboard(dashboardData: SavedDashboardCreate): Promise<SavedDashboard> {
    return this.apiCallWithBody<SavedDashboard>('/dashboard/saved', dashboardData)
  }

  async updateDashboardState(dashboardId: string, dashboardState: DashboardState): Promise<SavedDashboard> {
    return this.apiCallWithBodyPut<SavedDashboard>(
      `/dashboard/saved/${encodeURIComponent(dashboardId)}`,
      dashboardState
    )
  }

  async deleteSavedDashboard(dashboardId: string): Promise<void> {
    return this.apiCallDelete(`/dashboard/saved/${encodeURIComponent(dashboardId)}`)
  }

  // User methods
  async whoami(): Promise<{ user_email: string }> {
    return this.apiCall<{ user_email: string }>('/whoami')
  }

  // SQL query methods
  async listTables(): Promise<TableInfo[]> {
    return this.apiCall<TableInfo[]>('/sql/tables')
  }

  async getTableSchema(tableName: string): Promise<TableSchema> {
    return this.apiCall<TableSchema>(`/sql/tables/${encodeURIComponent(tableName)}/schema`)
  }

  async executeQuery(request: SQLQueryRequest): Promise<SQLQueryResponse> {
    return this.apiCallWithBody<SQLQueryResponse>('/sql/query', request)
  }

  async generateAIQuery(description: string): Promise<AIQueryResponse> {
    return this.apiCallWithBody<AIQueryResponse>('/sql/generate-query', {
      description,
    })
  }

  // Training run methods
  async getTrainingRuns(): Promise<TrainingRunListResponse> {
    return this.apiCall<TrainingRunListResponse>('/training-runs')
  }

  async getTrainingRun(runId: string): Promise<TrainingRun> {
    return this.apiCall<TrainingRun>(`/training-runs/${encodeURIComponent(runId)}`)
  }

  async updateTrainingRunDescription(runId: string, description: string): Promise<TrainingRun> {
    return this.apiCallWithBodyPut<TrainingRun>(`/training-runs/${encodeURIComponent(runId)}/description`, {
      description,
    })
  }

  async updateTrainingRunTags(runId: string, tags: string[]): Promise<TrainingRun> {
    return this.apiCallWithBodyPut<TrainingRun>(`/training-runs/${encodeURIComponent(runId)}/tags`, { tags })
  }

  async generateTrainingRunScorecard(
    runId: string,
    request: TrainingRunScorecardRequest
  ): Promise<PolicyScorecardData> {
    return this.apiCallWithBody<PolicyScorecardData>(`/scorecard/training-run/${encodeURIComponent(runId)}`, request)
  }

  async getTrainingRunPolicies(runId: string): Promise<TrainingRunPolicy[]> {
    return this.apiCall<TrainingRunPolicy[]>(`/training-runs/${encodeURIComponent(runId)}/policies`)
  }

  async createEvalTask(request: EvalTaskCreateRequest): Promise<EvalTask> {
    return this.apiCallWithBody<EvalTask>('/tasks', request)
  }

  async getEvalTasks(): Promise<EvalTask[]> {
    const response = await this.apiCall<EvalTasksResponse>('/tasks/all?limit=500')
    return response.tasks
  }

  async getPolicyIds(policyNames: string[]): Promise<Record<string, string>> {
    const params = new URLSearchParams()
    policyNames.forEach((name) => params.append('policy_names', name))
    const response = await this.apiCall<{ policy_ids: Record<string, string> }>(`/stats/policies/ids?${params}`)
    return response.policy_ids
  }

  // Policy-based scorecard methods
  async getPolicies(): Promise<PoliciesResponse> {
    return this.apiCall<PoliciesResponse>('/scorecard/policies')
  }

  async getEvalNames(request: EvalNamesRequest): Promise<Set<string>> {
    const res = await this.apiCallWithBody<string[]>('/scorecard/evals', request)
    return new Set(res)
  }

  async getAvailableMetrics(request: MetricsRequest): Promise<string[]> {
    return this.apiCallWithBody<string[]>('/scorecard/metrics', request)
  }

  async generatePolicyScorecard(request: PolicyScorecardRequest): Promise<PolicyScorecardData> {
    return this.apiCallWithBody<PolicyScorecardData>('/scorecard/scorecard', request)
  }

  // Leaderboard methods
  async listLeaderboards(): Promise<LeaderboardListResponse> {
    return this.apiCall<LeaderboardListResponse>('/leaderboards')
  }

  async getLeaderboard(leaderboardId: string): Promise<Leaderboard> {
    return this.apiCall<Leaderboard>(`/leaderboards/${encodeURIComponent(leaderboardId)}`)
  }

  async createLeaderboard(leaderboardData: LeaderboardCreateOrUpdate): Promise<Leaderboard> {
    return this.apiCallWithBody<Leaderboard>('/leaderboards', leaderboardData)
  }

  async updateLeaderboard(leaderboardId: string, leaderboardData: LeaderboardCreateOrUpdate): Promise<Leaderboard> {
    return this.apiCallWithBodyPut<Leaderboard>(`/leaderboards/${encodeURIComponent(leaderboardId)}`, leaderboardData)
  }

  async deleteLeaderboard(leaderboardId: string): Promise<void> {
    return this.apiCallDelete(`/leaderboards/${encodeURIComponent(leaderboardId)}`)
  }

  async generateLeaderboardScorecard(
    leaderboardId: string,
    request: LeaderboardScorecardRequest
  ): Promise<ScorecardData> {
    return this.apiCallWithBody<ScorecardData>('/scorecard/leaderboard', {
      leaderboard_id: leaderboardId,
      selector: request.selector,
      num_policies: request.num_policies,
    })
  }
}
