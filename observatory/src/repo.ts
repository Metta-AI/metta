import { DashboardData, PolicyEval, PolicyEvalMetric } from './data_loader';

export type HeatmapCell = {
  evalName: string;
  replayUrl: string | null;
  value: number;
  num_agents: number;
}

export type HeatmapData = {
  evalNames: Set<string>;
  cells : Map<string, Map<string, HeatmapCell>>;
  policyAverageScores: Map<string, number>;
  evalAverageScores: Map<string, number>;
  evalMaxScores: Map<string, number>;
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
  getGroupIds(suite: string): Promise<number[]>;

  getHeatmapData(metric: string, suite: string, groupId: number | null): Promise<HeatmapData>;
}

export class DataRepo implements Repo {
  constructor(private dashboardData: DashboardData) {
  }

  async getMetrics(suite: string): Promise<string[]> {
    return [...new Set(this.dashboardData.policy_evals.filter(row => row.suite === suite).flatMap(row => row.policy_eval_metrics.map(metric => metric.metric)))].sort();
  }

  async getSuites(): Promise<string[]> {
    return [...new Set(this.dashboardData.policy_evals.map(row => row.suite))].sort();
  }

  async getGroupIds(suite: string): Promise<number[]> {
    return [...new Set(this.dashboardData.policy_evals.filter(row => row.suite === suite).flatMap(row => row.policy_eval_metrics.map(metric => metric.group_id)))].sort();
  }

  async getHeatmapData(metric: string, suite: string, groupId: number | null): Promise<HeatmapData> {
    const evalNames = new Set<string>();
    const cells = new Map<string, Map<string, HeatmapCell>>();
    
    this.dashboardData.policy_evals.forEach((policyEval: PolicyEval) => {
      if (policyEval.suite === suite) {
        evalNames.add(policyEval.eval_name);
        const relevantMetrics = policyEval.policy_eval_metrics.filter((m: PolicyEvalMetric) => m.metric === metric && (groupId === null || m.group_id === groupId));
        if (relevantMetrics.length > 0) {
          let policyData = cells.get(policyEval.policy_uri);
          if (!policyData) {
            policyData = new Map<string, HeatmapCell>();
            cells.set(policyEval.policy_uri, policyData);
          }
          const cell = {
            evalName: policyEval.eval_name,
            replayUrl: policyEval.replay_url,
            value: 0,
            num_agents: 0
          }
          policyData.set(policyEval.eval_name, cell);

          relevantMetrics.forEach((m: PolicyEvalMetric) => {
            cell.value += m.sum_value;
            cell.num_agents += m.num_agents;
          });
        }
      }
    });

    const policyAverageScores = new Map<string, number>();
    cells.forEach((policyCells, policyUri) => {
      const policyAverageScore = Array.from(policyCells.values()).reduce((sum, cell) => sum + cell.value, 0) / policyCells.size;
      policyAverageScores.set(policyUri, policyAverageScore);
    });

    const evalAverageScores = new Map<string, number>();
    const evalMaxScores = new Map<string, number>();

    cells.forEach((policyCells, _) => {
      policyCells.forEach((cell, evalName) => {
        evalAverageScores.set(evalName, (evalAverageScores.get(evalName) || 0) + cell.value);
        evalMaxScores.set(evalName, Math.max(evalMaxScores.get(evalName) || 0, cell.value));
      });
    });
    evalAverageScores.forEach((value, key) => {
      evalAverageScores.set(key, value / evalNames.size);
    });

    return { evalNames, cells, policyAverageScores, evalAverageScores, evalMaxScores };
  }
}
