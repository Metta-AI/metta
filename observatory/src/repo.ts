export type HeatmapCell = {
  evalName: string
  replayUrl: string | null
  value: number
}

export type GroupDiff = {
  group_1: string
  group_2: string
}

export type PolicySelector = 'latest' | 'best'

export type GroupHeatmapMetric = GroupDiff | string

export type HeatmapData = {
  evalNames: string[]
  cells: Record<string, Record<string, HeatmapCell>>
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

export type SavedDashboard = {
  id: string
  name: string
  description: string | null
  type: string
  dashboard_state: Record<string, any>
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

export type Episode = {
  id: string
  created_at: string
  primary_policy_id: string
  eval_category: string | null
  env_name: string | null
  attributes: Record<string, any>
  // Policy information
  policy_name: string | null
  // Training run information
  training_run_id: string | null
  training_run_name: string | null
  training_run_user_id: string | null
  // Episode tags
  tags: string[]
}

export type EpisodeFilterResponse = {
  episodes: Episode[]
  total_count: number
  page: number
  page_size: number
  total_pages: number
}

export type EpisodeTagRequest = {
  episode_ids: string[]
  tag: string
}

export type EpisodeTagByFilterRequest = {
  filter_query: string
  tag: string
}

export type EpisodeTagResponse = {
  episodes_affected: number
}

export type AllTagsResponse = {
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
}

export type EvalTasksResponse = {
  tasks: EvalTask[]
}

// Policy-based heatmap types
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

export type PoliciesRequest = {
  search_text?: string
  pagination: PaginationRequest
}

export type PoliciesResponse = {
  policies: UnifiedPolicyInfo[]
  total_count: number
  page: number
  page_size: number
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

export type PolicyHeatmapRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
  eval_names: string[]
  training_run_policy_selector: 'latest' | 'best'
  metric: string
}

export type TrainingRunHeatmapRequest = {
  eval_names: string[]
  metric: string
}

export type TrainingRunPolicy = {
  policy_name: string
  policy_id: string
  epoch_start: number | null
  epoch_end: number | null
}

export type PolicyHeatmapCell = {
  evalName: string
  replayUrl: string | null
  value: number
}

export type PolicyHeatmapData = {
  evalNames: string[]
  policyNames: string[]
  cells: Record<string, Record<string, PolicyHeatmapCell>>
  policyAverageScores: Record<string, number>
  evalAverageScores: Record<string, number>
  evalMaxScores: Record<string, number>
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
  updateSavedDashboard(dashboardId: string, dashboardData: SavedDashboardCreate): Promise<SavedDashboard>
  deleteSavedDashboard(dashboardId: string): Promise<void>

  // User methods
  whoami(): Promise<{ user_email: string }>

  // SQL query methods
  listTables(): Promise<TableInfo[]>
  getTableSchema(tableName: string): Promise<TableSchema>
  executeQuery(request: SQLQueryRequest): Promise<SQLQueryResponse>

  // Training run methods
  getTrainingRuns(): Promise<TrainingRunListResponse>
  getTrainingRun(runId: string): Promise<TrainingRun>
  updateTrainingRunDescription(runId: string, description: string): Promise<TrainingRun>
  updateTrainingRunTags(runId: string, tags: string[]): Promise<TrainingRun>
  generateTrainingRunHeatmap(runId: string, request: TrainingRunHeatmapRequest): Promise<PolicyHeatmapData>
  getTrainingRunPolicies(runId: string): Promise<TrainingRunPolicy[]>

  // Episode methods
  filterEpisodes(page: number, pageSize: number, filterQuery: string): Promise<EpisodeFilterResponse>
  addEpisodeTags(episodeIds: string[], tag: string): Promise<EpisodeTagResponse>
  removeEpisodeTags(episodeIds: string[], tag: string): Promise<EpisodeTagResponse>
  addEpisodeTagsByFilter(filterQuery: string, tag: string): Promise<EpisodeTagResponse>
  removeEpisodeTagsByFilter(filterQuery: string, tag: string): Promise<EpisodeTagResponse>
  getAllEpisodeTags(): Promise<AllTagsResponse>

  // Eval task methods
  createEvalTask(request: EvalTaskCreateRequest): Promise<EvalTask>
  getEvalTasks(): Promise<EvalTask[]>

  // Policy methods
  getPolicyIds(policyNames: string[]): Promise<Record<string, string>>

  // Policy-based heatmap methods
  getPolicies(request: PoliciesRequest): Promise<PoliciesResponse>
  getEvalNames(request: EvalNamesRequest): Promise<Set<string>>
  getAvailableMetrics(request: MetricsRequest): Promise<string[]>
  generatePolicyHeatmap(request: PolicyHeatmapRequest): Promise<PolicyHeatmapData>

}

export class ServerRepo implements Repo {
  constructor(private baseUrl: string = 'http://localhost:8000') { }

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
      headers: this.getHeaders()
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
      headers: this.getHeaders()
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

  async updateSavedDashboard(dashboardId: string, dashboardData: SavedDashboardCreate): Promise<SavedDashboard> {
    return this.apiCallWithBodyPut<SavedDashboard>(`/dashboard/saved/${encodeURIComponent(dashboardId)}`, dashboardData)
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

  // Training run methods
  async getTrainingRuns(): Promise<TrainingRunListResponse> {
    return this.apiCall<TrainingRunListResponse>('/dashboard/training-runs')
  }

  async getTrainingRun(runId: string): Promise<TrainingRun> {
    return this.apiCall<TrainingRun>(`/dashboard/training-runs/${encodeURIComponent(runId)}`)
  }

  async updateTrainingRunDescription(runId: string, description: string): Promise<TrainingRun> {
    return this.apiCallWithBodyPut<TrainingRun>(`/dashboard/training-runs/${encodeURIComponent(runId)}/description`, {
      description,
    })
  }

  async updateTrainingRunTags(runId: string, tags: string[]): Promise<TrainingRun> {
    return this.apiCallWithBodyPut<TrainingRun>(`/dashboard/training-runs/${encodeURIComponent(runId)}/tags`, { tags })
  }

  async generateTrainingRunHeatmap(runId: string, request: TrainingRunHeatmapRequest): Promise<PolicyHeatmapData> {
    return this.apiCallWithBody<PolicyHeatmapData>(`/heatmap/training-run/${encodeURIComponent(runId)}`, request)
  }

  async getTrainingRunPolicies(runId: string): Promise<TrainingRunPolicy[]> {
    return this.apiCall<TrainingRunPolicy[]>(`/heatmap/training-run/${encodeURIComponent(runId)}/policies`)
  }

  // Episode methods
  async filterEpisodes(page: number, pageSize: number, filterQuery: string): Promise<EpisodeFilterResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      filter_query: filterQuery,
    })
    return this.apiCall<EpisodeFilterResponse>(`/episodes?${params}`)
  }

  async addEpisodeTags(episodeIds: string[], tag: string): Promise<EpisodeTagResponse> {
    return this.apiCallWithBody<EpisodeTagResponse>('/episodes/tags/add', {
      episode_ids: episodeIds,
      tag: tag,
    })
  }

  async removeEpisodeTags(episodeIds: string[], tag: string): Promise<EpisodeTagResponse> {
    return this.apiCallWithBody<EpisodeTagResponse>('/episodes/tags/remove', {
      episode_ids: episodeIds,
      tag: tag,
    })
  }

  async addEpisodeTagsByFilter(filterQuery: string, tag: string): Promise<EpisodeTagResponse> {
    return this.apiCallWithBody<EpisodeTagResponse>('/episodes/tags/add-by-filter', {
      filter_query: filterQuery,
      tag: tag,
    })
  }

  async removeEpisodeTagsByFilter(filterQuery: string, tag: string): Promise<EpisodeTagResponse> {
    return this.apiCallWithBody<EpisodeTagResponse>('/episodes/tags/remove-by-filter', {
      filter_query: filterQuery,
      tag: tag,
    })
  }

  async getAllEpisodeTags(): Promise<AllTagsResponse> {
    return this.apiCall<AllTagsResponse>('/episodes/tags/all')
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
    policyNames.forEach(name => params.append('policy_names', name))
    const response = await this.apiCall<{ policy_ids: Record<string, string> }>(`/stats/policies/ids?${params}`)
    return response.policy_ids
  }


  // Policy-based heatmap methods
  async getPolicies(request: PoliciesRequest): Promise<PoliciesResponse> {
    return this.apiCallWithBody<PoliciesResponse>('/heatmap/policies', request)
  }

  async getEvalNames(request: EvalNamesRequest): Promise<Set<string>> {
    const res = await this.apiCallWithBody<string[]>('/heatmap/evals', request)
    return new Set(res)
  }

  async getAvailableMetrics(request: MetricsRequest): Promise<string[]> {
    return this.apiCallWithBody<string[]>('/heatmap/metrics', request)
  }

  async generatePolicyHeatmap(request: PolicyHeatmapRequest): Promise<PolicyHeatmapData> {
    return this.apiCallWithBody<PolicyHeatmapData>('/heatmap/heatmap', request)
  }
}
