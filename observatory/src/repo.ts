import type { components } from './api-types'
import { getToken, initiateLogin } from './auth'
import type {
  TaskFilters,
  TokenCreate,
  TokenListResponse,
  TokenResponse,
  TrainingRun,
  TrainingRunListResponse,
  TrainingRunPolicy,
  ValueOverReplacementSummary,
} from './client-types'

type Schemas = components['schemas']

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

  private async apiCall<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: this.getHeaders(),
    })
    if (!response.ok) {
      if (response.status === 401) {
        initiateLogin()
        throw new Error('Unauthorized - redirecting to login')
      }
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallWithBody<T>(endpoint: string, body: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: this.getHeaders('application/json'),
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      if (response.status === 401) {
        initiateLogin()
        throw new Error('Unauthorized - redirecting to login')
      }
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallWithBodyPut<T>(endpoint: string, body: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'PUT',
      headers: this.getHeaders('application/json'),
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      if (response.status === 401) {
        initiateLogin()
        throw new Error('Unauthorized - redirecting to login')
      }
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
      if (response.status === 401) {
        initiateLogin()
        throw new Error('Unauthorized - redirecting to login')
      }
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

  // User methods
  async whoami(): Promise<Schemas['WhoAmIResponse']> {
    return this.apiCall<Schemas['WhoAmIResponse']>('/whoami')
  }

  // SQL query methods
  async listTables(): Promise<Schemas['TableInfo'][]> {
    return this.apiCall<Schemas['TableInfo'][]>('/sql/tables')
  }

  async getTableSchema(tableName: string): Promise<Schemas['TableSchema']> {
    return this.apiCall<Schemas['TableSchema']>(`/sql/tables/${encodeURIComponent(tableName)}/schema`)
  }

  async executeQuery(request: Schemas['SQLQueryRequest']): Promise<Schemas['SQLQueryResponse']> {
    return this.apiCallWithBody<Schemas['SQLQueryResponse']>('/sql/query', request)
  }

  async generateAIQuery(description: string): Promise<Schemas['AIQueryResponse']> {
    return this.apiCallWithBody<Schemas['AIQueryResponse']>('/sql/generate-query', {
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

  async createEvalTask(request: Schemas['TaskCreateRequest']): Promise<Schemas['EvalTaskRow']> {
    return this.apiCallWithBody<Schemas['EvalTaskRow']>('/tasks', request)
  }

  async getEvalTasksPaginated(
    page: number,
    pageSize: number,
    filters: TaskFilters
  ): Promise<Schemas['PaginatedTasksResponse']> {
    const params = new URLSearchParams()
    params.append('page', page.toString())
    params.append('page_size', pageSize.toString())

    if (filters.command?.trim()) params.append('command', filters.command.trim())
    if (filters.user_id?.trim()) params.append('user_id', filters.user_id.trim())
    if (filters.status?.trim()) params.append('status', filters.status.trim())
    if (filters.assignee?.trim()) params.append('assignee', filters.assignee.trim())
    if (filters.git_hash?.trim()) params.append('git_hash', filters.git_hash.trim())
    if (filters.created_at?.trim()) params.append('created_at', filters.created_at.trim())
    if (filters.assigned_at?.trim()) params.append('assigned_at', filters.assigned_at.trim())

    return this.apiCall<Schemas['PaginatedTasksResponse']>(`/tasks/paginated?${params}`)
  }

  async getEvalTask(taskId: number): Promise<Schemas['EvalTaskRow']> {
    return this.apiCall<Schemas['EvalTaskRow']>(`/tasks/${taskId}`)
  }

  async getTaskAttempts(taskId: number): Promise<Schemas['TaskAttemptsResponse']> {
    return this.apiCall<Schemas['TaskAttemptsResponse']>(`/tasks/${taskId}/attempts`)
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
  async getPublicLeaderboard(): Promise<Schemas['LeaderboardPoliciesResponse']> {
    return this.apiCall<Schemas['LeaderboardPoliciesResponse']>('/leaderboard/v2')
  }

  async getPublicLeaderboardWithVor(): Promise<Schemas['LeaderboardPoliciesResponse']> {
    return this.apiCall<Schemas['LeaderboardPoliciesResponse']>('/leaderboard/v2/vor')
  }

  async getPersonalLeaderboard(): Promise<Schemas['LeaderboardPoliciesResponse']> {
    return this.apiCall<Schemas['LeaderboardPoliciesResponse']>('/leaderboard/v2/users/me')
  }

  async getLeaderboardPolicy(policyVersionId: string): Promise<Schemas['LeaderboardPoliciesResponse']> {
    return this.apiCall<Schemas['LeaderboardPoliciesResponse']>(`/leaderboard/v2/policy/${policyVersionId}`)
  }

  async getValueOverReplacementDetail(policyVersionId: string): Promise<ValueOverReplacementSummary | null> {
    try {
      return await this.apiCall<ValueOverReplacementSummary>(`/leaderboard/v2/vor/${policyVersionId}`)
    } catch {
      return null
    }
  }

  async getPolicyVersion(policyVersionId: string): Promise<Schemas['PolicyVersionWithName']> {
    return this.apiCall<Schemas['PolicyVersionWithName']>(`/stats/policies/versions/${policyVersionId}`)
  }

  async getPolicyVersionsBatch(policyVersionIds: string[]): Promise<Schemas['PublicPolicyVersionRow'][]> {
    const chunkSize = 10
    const results: Schemas['PublicPolicyVersionRow'][] = []

    for (let i = 0; i < policyVersionIds.length; i += chunkSize) {
      const chunk = policyVersionIds.slice(i, i + chunkSize)
      const params = chunk.map((id) => `policy_version_ids=${id}`).join('&')
      const response = await this.apiCall<Schemas['PolicyVersionsResponse']>(
        `/stats/policy-versions?${params}&limit=${chunk.length}`
      )
      results.push(...response.entries)
    }

    return results
  }

  async queryEpisodes(request: Schemas['EpisodeQueryRequest']): Promise<Schemas['EpisodeQueryResponse']> {
    return this.apiCallWithBody<Schemas['EpisodeQueryResponse']>('/stats/episodes/query', request)
  }

  async getPolicies(params?: {
    name_exact?: string
    name_fuzzy?: string
    limit?: number
    offset?: number
  }): Promise<Schemas['PoliciesResponse']> {
    const searchParams = new URLSearchParams()
    if (params?.name_exact) searchParams.append('name_exact', params.name_exact)
    if (params?.name_fuzzy) searchParams.append('name_fuzzy', params.name_fuzzy)
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<Schemas['PoliciesResponse']>(`/stats/policies${query ? `?${query}` : ''}`)
  }

  async getPolicyVersions(params?: {
    name_exact?: string
    name_fuzzy?: string
    limit?: number
    offset?: number
  }): Promise<Schemas['PolicyVersionsResponse']> {
    const searchParams = new URLSearchParams()
    if (params?.name_exact) searchParams.append('name_exact', params.name_exact)
    if (params?.name_fuzzy) searchParams.append('name_fuzzy', params.name_fuzzy)
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<Schemas['PolicyVersionsResponse']>(`/stats/policy-versions${query ? `?${query}` : ''}`)
  }

  async getVersionsForPolicy(
    policyId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<Schemas['PolicyVersionsResponse']> {
    const searchParams = new URLSearchParams()
    if (params?.limit !== undefined) searchParams.append('limit', params.limit.toString())
    if (params?.offset !== undefined) searchParams.append('offset', params.offset.toString())
    const query = searchParams.toString()
    return this.apiCall<Schemas['PolicyVersionsResponse']>(
      `/stats/policies/${policyId}/versions${query ? `?${query}` : ''}`
    )
  }
}
