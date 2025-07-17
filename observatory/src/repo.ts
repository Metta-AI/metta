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
  getSuites(): Promise<string[]>
  getMetrics(suite: string): Promise<string[]>
  getAllMetrics(): Promise<string[]>
  getGroupIds(suite: string): Promise<string[]>

  getHeatmapData(
    metric: string,
    suite: string,
    policySelector?: PolicySelector
  ): Promise<HeatmapData>

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
  getTrainingRunHeatmapData(
    runId: string,
    metric: string,
    suite: string,
  ): Promise<HeatmapData>
}

export class ServerRepo implements Repo {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  private async apiCall<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`)
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
    return response.json()
  }

  private async apiCallWithBody<T>(endpoint: string, body: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
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
      headers: {
        'Content-Type': 'application/json',
      },
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
    })
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`)
    }
  }

  async getSuites(): Promise<string[]> {
    return this.apiCall<string[]>('/dashboard/suites')
  }

  async getMetrics(suite: string): Promise<string[]> {
    return this.apiCall<string[]>(`/dashboard/suites/${encodeURIComponent(suite)}/metrics`)
  }

  async getAllMetrics(): Promise<string[]> {
    return this.apiCall<string[]>('/dashboard/metrics')
  }

  async getGroupIds(suite: string): Promise<string[]> {
    return this.apiCall<string[]>(`/dashboard/suites/${encodeURIComponent(suite)}/group-ids`)
  }

  async getHeatmapData(
    metric: string,
    suite: string,
    policySelector: PolicySelector = 'latest'
  ): Promise<HeatmapData> {
    // Use POST endpoint for GroupDiff
    const apiData = await this.apiCallWithBody<HeatmapData>(
      `/dashboard/suites/${encodeURIComponent(suite)}/metrics/${encodeURIComponent(metric)}/heatmap`,
      { policy_selector: policySelector }
    )
    return apiData
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

  async getTrainingRunHeatmapData(
    runId: string,
    metric: string,
    suite: string,
  ): Promise<HeatmapData> {
    return this.apiCall<HeatmapData>(
      `/dashboard/training-runs/${encodeURIComponent(runId)}/suites/${encodeURIComponent(suite)}/metrics/${encodeURIComponent(metric)}/heatmap`,
    )
  }
}
