import { getToken, initiateLogin } from './auth'

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

export type TrainingRunDescriptionUpdate = {
  description: string
}

export type TrainingRunTagsUpdate = {
  tags: string[]
}

export type EvalTaskCreateRequest = {
  command: string
  git_hash: string | null
  attributes: Record<string, any>
}

export type TaskStatus = 'unprocessed' | 'running' | 'canceled' | 'done' | 'error' | 'system_error'

type TaskStatusMixin = {
  status: TaskStatus
  status_details: Record<string, any> | null
}

export type EvalTask = {
  // eval_tasks table columns
  id: number
  command: string
  data_uri: string | null
  git_hash: string | null
  attributes: Record<string, any>
  user_id: string
  created_at: string
  is_finished: boolean
  latest_attempt_id: number | null

  // Latest attempt columns (from JOIN)
  attempt_number: number | null
  assigned_at: string | null
  assignee: string | null
  started_at: string | null
  finished_at: string | null
  output_log_path: string | null
} & TaskStatusMixin

export type TaskAttempt = {
  id: number
  task_id: number
  attempt_number: number
  assigned_at: string | null
  assignee: string | null
  started_at: string | null
  finished_at: string | null
  output_log_path: string | null
} & TaskStatusMixin

export type PaginatedEvalTasksResponse = {
  tasks: EvalTask[]
  total_count: number
  page: number
  page_size: number
  total_pages: number
}

export type TaskAttemptsResponse = {
  attempts: TaskAttempt[]
}

export type TaskFilters = {
  command?: string
  user_id?: string
  status?: string
  assignee?: string
  git_hash?: string
  created_at?: string
  assigned_at?: string
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

export type EvalNamesRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
}

export type MetricsRequest = {
  training_run_ids: string[]
  run_free_policy_ids: string[]
  eval_names: string[]
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

export type PublicPolicyVersionRow = {
  id: string
  policy_id: string
  created_at: string
  policy_created_at: string
  user_id: string
  name: string
  version: number
  tags: Record<string, string>
  version_count?: number
}

export type EpisodeReplay = {
  episode_id: string
  replay_url: string
}

export type EpisodeWithTags = {
  id: string
  primary_pv_id: string | null
  replay_url: string | null
  thumbnail_url: string | null
  attributes: Record<string, any>
  eval_task_id: string | null
  created_at: string
  tags: Record<string, string>
  avg_rewards: Record<string, number>
}

export type LeaderboardPolicyEntry = {
  policy_version: PublicPolicyVersionRow
  scores: Record<string, number>
  avg_score: number | null
  overall_vor: number | null
  replays: Record<string, EpisodeReplay[]>
  score_episode_ids: Record<string, string | null>
}

export type LeaderboardPoliciesResponse = {
  entries: LeaderboardPolicyEntry[]
}

export type ValueOverReplacementSummary = {
  policy_version_id: string
  overall_vor: number | null
  overall_vor_std: number | null
  total_candidate_agents: number
}

export type PolicyVersionWithName = {
  id: string
  policy_id: string
  version: number
  name: string
  created_at: string
}

export type EpisodeQueryRequest = {
  primary_policy_version_ids?: string[]
  tag_filters?: Record<string, string[] | null>
  limit?: number | null
  offset?: number
  episode_ids?: string[]
}

export type EpisodeQueryResponse = {
  episodes: EpisodeWithTags[]
}

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

export type JobStatus = 'pending' | 'dispatched' | 'running' | 'completed' | 'failed'

export type MatchStatus = 'pending' | 'scheduled' | 'running' | 'completed' | 'failed'

export type SeasonDetail = {
  name: string
  pools: string[]
}

export type LeaderboardEntry = {
  rank: number
  policy_version_id: string
  policy_name: string | null
  policy_version: number | null
  score: number
  matches: number
}

export type LeaderboardResponse = {
  entries: LeaderboardEntry[]
}

export type SubmissionResponse = {
  pools: string[]
}

export type PoolMembership = {
  pool_name: string
  active: boolean
}

export type PolicySummary = {
  policy_version_id: string
  policy_name: string | null
  policy_version: number | null
  pools: PoolMembership[]
}

export type SeasonMatchPlayerSummary = {
  policy_version_id: string
  policy_name: string | null
  policy_version: number | null
  policy_index: number
  score: number | null
}

export type SeasonMatchSummary = {
  id: string
  pool_name: string
  status: MatchStatus
  assignments: number[]
  players: SeasonMatchPlayerSummary[]
  episode_id: string | null
  created_at: string
}

export type MembershipHistoryEntry = {
  pool_name: string
  action: string
  notes: string | null
  created_at: string
}

export type PlayerDetail = {
  policy_version_id: string
  policy_name: string | null
  policy_version: number | null
  membership_history: MembershipHistoryEntry[]
}

export type JobRequest = {
  id: string
  job_type: string
  job: Record<string, any>
  status: JobStatus
  user_id: string
  worker: string | null
  result: Record<string, any> | null
  created_at: string
  dispatched_at: string | null
  running_at: string | null
  completed_at: string | null
}

export type PolicyRow = {
  id: string
  name: string
  created_at: string
  user_id: string
  attributes: Record<string, any>
  version_count: number
}

export type PoliciesResponse = {
  entries: PolicyRow[]
  total_count: number
}

export type PolicyVersionsResponse = {
  entries: PublicPolicyVersionRow[]
  total_count: number
}

export class Repo {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  private getHeaders(contentType?: string): Record<string, string> {
    const headers: Record<string, string> = {}

    if (contentType) {
      headers['Content-Type'] = contentType
    }

    const token = getToken()
    if (token) {
      headers['X-Auth-Token'] = token
    }

    return headers
  }

  private async handleErrorResponse(response: Response): Promise<never> {
    if (response.status === 401) {
      initiateLogin()
      throw new Error('Unauthorized - redirecting to login')
    }
    // Try to extract error detail from response body
    try {
      const body = await response.json()
      if (body.detail) {
        throw new Error(body.detail)
      }
    } catch {
      // Ignore JSON parse errors, fall through to default
    }
    throw new Error(`API call failed: ${response.status} ${response.statusText}`)
  }

  private async apiCall<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: this.getHeaders(),
    })
    if (!response.ok) {
      await this.handleErrorResponse(response)
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
      await this.handleErrorResponse(response)
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
      await this.handleErrorResponse(response)
    }
    return response.json()
  }

  private async apiCallDelete(endpoint: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    })
    if (!response.ok) {
      await this.handleErrorResponse(response)
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

  async getTrainingRunPolicies(runId: string): Promise<TrainingRunPolicy[]> {
    return this.apiCall<TrainingRunPolicy[]>(`/training-runs/${encodeURIComponent(runId)}/policies`)
  }

  async createEvalTask(request: EvalTaskCreateRequest): Promise<EvalTask> {
    return this.apiCallWithBody<EvalTask>('/tasks', request)
  }

  async getEvalTasksPaginated(
    page: number,
    pageSize: number,
    filters: TaskFilters
  ): Promise<PaginatedEvalTasksResponse> {
    const params = new URLSearchParams()
    params.append('page', page.toString())
    params.append('page_size', pageSize.toString())

    // Only append non-empty filter values
    if (filters.command?.trim()) params.append('command', filters.command.trim())
    if (filters.user_id?.trim()) params.append('user_id', filters.user_id.trim())
    if (filters.status?.trim()) params.append('status', filters.status.trim())
    if (filters.assignee?.trim()) params.append('assignee', filters.assignee.trim())
    if (filters.git_hash?.trim()) params.append('git_hash', filters.git_hash.trim())
    if (filters.created_at?.trim()) params.append('created_at', filters.created_at.trim())
    if (filters.assigned_at?.trim()) params.append('assigned_at', filters.assigned_at.trim())

    return this.apiCall<PaginatedEvalTasksResponse>(`/tasks/paginated?${params}`)
  }

  async getEvalTask(taskId: number): Promise<EvalTask> {
    return this.apiCall<EvalTask>(`/tasks/${taskId}`)
  }

  async getTaskAttempts(taskId: number): Promise<TaskAttemptsResponse> {
    return this.apiCall<TaskAttemptsResponse>(`/tasks/${taskId}/attempts`)
  }

  getTaskLogUrl(taskId: number, logType: 'output'): string {
    return `${this.baseUrl}/tasks/${taskId}/logs/${logType}`
  }

  // Policy methods
  async getPolicyIds(policyNames: string[]): Promise<Record<string, string>> {
    const params = new URLSearchParams()
    policyNames.forEach((name) => params.append('policy_names', name))
    const response = await this.apiCall<{ policy_ids: Record<string, string> }>(`/stats/policies/ids?${params}`)
    return response.policy_ids
  }

  // Leaderboard / policy version queries
  async getPublicLeaderboard(): Promise<LeaderboardPoliciesResponse> {
    return this.apiCall<LeaderboardPoliciesResponse>('/leaderboard/v2')
  }

  async getPublicLeaderboardWithVor(): Promise<LeaderboardPoliciesResponse> {
    return this.apiCall<LeaderboardPoliciesResponse>('/leaderboard/v2/vor')
  }

  async getPersonalLeaderboard(): Promise<LeaderboardPoliciesResponse> {
    return this.apiCall<LeaderboardPoliciesResponse>('/leaderboard/v2/users/me')
  }

  async getLeaderboardPolicy(policyVersionId: string): Promise<LeaderboardPoliciesResponse> {
    return this.apiCall<LeaderboardPoliciesResponse>(`/leaderboard/v2/policy/${policyVersionId}`)
  }

  async getValueOverReplacementDetail(policyVersionId: string): Promise<ValueOverReplacementSummary | null> {
    try {
      return await this.apiCall<ValueOverReplacementSummary>(`/leaderboard/v2/vor/${policyVersionId}`)
    } catch {
      return null
    }
  }

  async getPolicyVersion(policyVersionId: string): Promise<PolicyVersionWithName> {
    return this.apiCall<PolicyVersionWithName>(`/stats/policies/versions/${policyVersionId}`)
  }

  async getPolicyVersionsBatch(policyVersionIds: string[]): Promise<PublicPolicyVersionRow[]> {
    const chunkSize = 10
    const results: PublicPolicyVersionRow[] = []

    for (let i = 0; i < policyVersionIds.length; i += chunkSize) {
      const chunk = policyVersionIds.slice(i, i + chunkSize)
      const params = chunk.map((id) => `policy_version_ids=${id}`).join('&')
      const response = await this.apiCall<PolicyVersionsResponse>(
        `/stats/policy-versions?${params}&limit=${chunk.length}`
      )
      results.push(...response.entries)
    }

    return results
  }

  async queryEpisodes(request: EpisodeQueryRequest): Promise<EpisodeQueryResponse> {
    return this.apiCallWithBody<EpisodeQueryResponse>('/stats/episodes/query', request)
  }

  async getPolicies(params?: {
    name_exact?: string
    name_fuzzy?: string
    limit?: number
    offset?: number
  }): Promise<PoliciesResponse> {
    const searchParams = new URLSearchParams()
    if (params?.name_exact) searchParams.append('name_exact', params.name_exact)
    if (params?.name_fuzzy) searchParams.append('name_fuzzy', params.name_fuzzy)
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<PoliciesResponse>(`/stats/policies${query ? `?${query}` : ''}`)
  }

  async getPolicyVersions(params?: {
    name_exact?: string
    name_fuzzy?: string
    limit?: number
    offset?: number
  }): Promise<PolicyVersionsResponse> {
    const searchParams = new URLSearchParams()
    if (params?.name_exact) searchParams.append('name_exact', params.name_exact)
    if (params?.name_fuzzy) searchParams.append('name_fuzzy', params.name_fuzzy)
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<PolicyVersionsResponse>(`/stats/policy-versions${query ? `?${query}` : ''}`)
  }

  async getVersionsForPolicy(
    policyId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<PolicyVersionsResponse> {
    const searchParams = new URLSearchParams()
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<PolicyVersionsResponse>(`/stats/policies/${policyId}/versions${query ? `?${query}` : ''}`)
  }

  async getJobs(params?: {
    job_type?: string
    statuses?: JobStatus[]
    limit?: number
    offset?: number
  }): Promise<JobRequest[]> {
    const searchParams = new URLSearchParams()
    if (params?.job_type) searchParams.append('job_type', params.job_type)
    if (params?.statuses) {
      for (const status of params.statuses) {
        searchParams.append('statuses', status)
      }
    }
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<JobRequest[]>(`/jobs${query ? `?${query}` : ''}`)
  }

  // Tournament methods
  async getSeasons(): Promise<string[]> {
    return this.apiCall<string[]>('/tournament/seasons')
  }

  async getSeason(seasonName: string): Promise<SeasonDetail> {
    return this.apiCall<SeasonDetail>(`/tournament/seasons/${encodeURIComponent(seasonName)}`)
  }

  async getSeasonLeaderboard(seasonName: string): Promise<LeaderboardEntry[]> {
    return this.apiCall<LeaderboardEntry[]>(`/tournament/seasons/${encodeURIComponent(seasonName)}/leaderboard`)
  }

  async getSeasonPolicies(seasonName: string): Promise<PolicySummary[]> {
    return this.apiCall<PolicySummary[]>(`/tournament/seasons/${encodeURIComponent(seasonName)}/policies`)
  }

  async getSeasonMatches(seasonName: string, params?: { limit?: number }): Promise<SeasonMatchSummary[]> {
    const query = params?.limit !== undefined ? `?limit=${params.limit}` : ''
    return this.apiCall<SeasonMatchSummary[]>(
      `/tournament/seasons/${encodeURIComponent(seasonName)}/matches${query}`
    )
  }

  async submitToSeason(seasonName: string, policyVersionId: string): Promise<SubmissionResponse> {
    return this.apiCallWithBody<SubmissionResponse>(`/tournament/seasons/${encodeURIComponent(seasonName)}/submit`, {
      policy_version_id: policyVersionId,
    })
  }

  async getSeasonPlayer(seasonName: string, policyVersionId: string): Promise<PlayerDetail> {
    return this.apiCall<PlayerDetail>(
      `/tournament/seasons/${encodeURIComponent(seasonName)}/players/${encodeURIComponent(policyVersionId)}`
    )
  }
}
