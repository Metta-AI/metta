export type PolicyEvalMetric = {
  policy_uri: string;
  eval_name: string;
  suite: string;
  metric: string;
  value: number;
  replay_url: string | null;
}

export type DashboardData = {
  policy_eval_metrics: PolicyEvalMetric[];
}

export async function loadDataFromUri(uri: string): Promise<DashboardData> {
  const response = await fetch(uri);
  return response.json();
}

export async function loadDataFromFile(file: File): Promise<DashboardData> {
  const text = await file.text();
  return JSON.parse(text);
}
