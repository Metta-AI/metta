import { PolicyEvalMetric, DashboardData } from './data_loader';

export interface Repo {
  getPolicyEvals(metric: string): Promise<PolicyEvalMetric[]>;
  getMetrics(): Promise<string[]>;
}

export class DataRepo implements Repo {
  constructor(private data: DashboardData) {}

  async getPolicyEvals(metric: string): Promise<PolicyEvalMetric[]> {
    return this.data.policy_eval_metrics.filter(row => row.metric === metric);
  }

  async getMetrics(): Promise<string[]> {
    return [...new Set(this.data.policy_eval_metrics.map(row => row.metric))].sort();
  }
}
