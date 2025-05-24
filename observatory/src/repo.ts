import { DashboardData } from './data_loader';

export type HeatmapCell = {
  evalName: string;
  replayUrl: string | null;
  value: number;
}

export type HeatmapData = {
  evalNames: Set<string>;
  cells : Map<string, Map<string, HeatmapCell>>;
  policyAverageScores: Map<string, number>;
  evalAverageScores: Map<string, number>;
  evalMaxScores: Map<string, number>;
}

export interface Repo {  
  getMetrics(): Promise<string[]>;
  getSuites(): Promise<string[]>;

  getHeatmapData(metric: string, suite: string): Promise<HeatmapData>;
}

export class DataRepo implements Repo {
  constructor(private dashboardData: DashboardData) {
  }

  async getMetrics(): Promise<string[]> {
    return [...new Set(this.dashboardData.policy_eval_metrics.map(row => row.metric))].sort();
  }

  async getSuites(): Promise<string[]> {
    return [...new Set(this.dashboardData.policy_eval_metrics.map(row => row.suite))].sort();
  }

  async getHeatmapData(metric: string, suite: string): Promise<HeatmapData> {
    const evalNames = new Set<string>();
    const cells = new Map<string, Map<string, HeatmapCell>>();
    this.dashboardData.policy_eval_metrics.forEach(row => {
      if (row.suite === suite) {
        evalNames.add(row.eval_name);
        if (row.metric === metric) {
          let policyData = cells.get(row.policy_uri);
          if (!policyData) {
            policyData = new Map<string, HeatmapCell>();
            cells.set(row.policy_uri, policyData);
          }
          policyData.set(row.eval_name, { evalName: row.eval_name, replayUrl: row.replay_url, value: row.value });
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
