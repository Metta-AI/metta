export type PolicyEvalMetric = {
  metric: string
  group_id: string
  sum_value: number
}

export type PolicyEval = {
  policy_uri: string
  eval_name: string
  suite: string
  replay_url: string | null
  group_num_agents: Record<string, number>
  policy_eval_metrics: Array<PolicyEvalMetric>
}

export type DashboardData = {
  policy_evals: Array<PolicyEval>
}

export async function loadDataFromUri(uri: string): Promise<DashboardData> {
  const response = await fetch(uri)
  return response.json()
}

export async function loadDataFromFile(file: File): Promise<DashboardData> {
  const text = await file.text()
  return JSON.parse(text)
}
