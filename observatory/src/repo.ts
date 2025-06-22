export type HeatmapCell = {
  evalName: string;
  replayUrl: string | null;
  value: number;
}

export type GroupDiff = {
  group_1: string;
  group_2: string;
}

export type GroupHeatmapMetric = GroupDiff | string

export type HeatmapData = {
  evalNames: string[];
  cells: Record<string, Record<string, HeatmapCell>>;
  policyAverageScores: Record<string, number>;
  evalAverageScores: Record<string, number>;
  evalMaxScores: Record<string, number>;
}

export type TokenInfo = {
  id: string;
  name: string;
  created_at: string;
  expiration_time: string;
  last_used_at: string | null;
}

export type TokenCreate = {
  name: string;
}

export type TokenResponse = {
  token: string;
}

export type TokenListResponse = {
  tokens: TokenInfo[];
}

/**
 * Interface for data fetching.
 *
 * Currently the data is loaded from a pre-computed JSON file.
 * In the future, we will fetch the data from an API.
 */
export interface Repo {
  getSuites(): Promise<string[]>;
  getMetrics(suite: string): Promise<string[]>;
  getGroupIds(suite: string): Promise<string[]>;

  getHeatmapData(metric: string, suite: string, groupMetric: GroupHeatmapMetric): Promise<HeatmapData>;

  // Token management methods
  createToken(tokenData: TokenCreate): Promise<TokenResponse>;
  listTokens(): Promise<TokenListResponse>;
  deleteToken(tokenId: string): Promise<void>;
}

export class ServerRepo implements Repo {
  constructor(private baseUrl: string = "http://localhost:8000") {
  }

  private async apiCall<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`);
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  private async apiCallWithBody<T>(endpoint: string, body: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  private async apiCallDelete(endpoint: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }
  }

  async getSuites(): Promise<string[]> {
    return this.apiCall<string[]>("/dashboard/suites");
  }

  async getMetrics(suite: string): Promise<string[]> {
    return this.apiCall<string[]>(`/dashboard/suites/${encodeURIComponent(suite)}/metrics`);
  }

  async getGroupIds(suite: string): Promise<string[]> {
    return this.apiCall<string[]>(`/dashboard/suites/${encodeURIComponent(suite)}/group-ids`);
  }

  async getHeatmapData(metric: string, suite: string, groupMetric: GroupHeatmapMetric): Promise<HeatmapData> {
    // Use POST endpoint for GroupDiff
    const apiData = await this.apiCallWithBody<HeatmapData>(
      `/dashboard/suites/${encodeURIComponent(suite)}/metrics/${encodeURIComponent(metric)}/heatmap`,
      { group_metric: groupMetric }
    );
    return apiData;
  }

  // Token management methods
  async createToken(tokenData: TokenCreate): Promise<TokenResponse> {
    return this.apiCallWithBody<TokenResponse>("/tokens", tokenData);
  }

  async listTokens(): Promise<TokenListResponse> {
    return this.apiCall<TokenListResponse>("/tokens");
  }

  async deleteToken(tokenId: string): Promise<void> {
    return this.apiCallDelete(`/tokens/${tokenId}`);
  }
}

