export type PolicyEvalMetric = {
  metric: string;
  group_id: number;
  sum_value: number;
  num_agents: number;
}

export type PolicyEval = {
  policy_uri: string;
  eval_name: string;
  suite: string;
  replay_url: string | null;
  policy_eval_metrics: PolicyEvalMetric[];
}

export type DashboardData = {
  policy_evals: PolicyEval[];
}

export async function loadDataFromUri(uri: string): Promise<DashboardData> {
  const response = await fetch(uri);
  return response.json();
}

export async function loadDataFromFile(file: File): Promise<DashboardData> {
  const text = await file.text();
  return JSON.parse(text);
}
