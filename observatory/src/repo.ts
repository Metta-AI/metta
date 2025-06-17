import { DashboardData, PolicyEval, PolicyEvalMetric } from './data_loader'

export type HeatmapCell = {
  evalName: string
  replayUrl: string | null
  value: number
}

export type HeatmapData = {
  evalNames: Set<string>
  cells: Map<string, Map<string, HeatmapCell>>
  policyAverageScores: Map<string, number>
  evalAverageScores: Map<string, number>
  evalMaxScores: Map<string, number>
}

export type GroupDiff = {
  group_1: string
  group_2: string
}

export type GroupHeatmapMetric = GroupDiff | string

/**
 * Interface for data fetching.
 *
 * Currently the data is loaded from a pre-computed JSON file.
 * In the future, we will fetch the data from an API.
 */
export interface Repo {
  getSuites(): Promise<string[]>
  getMetrics(suite: string): Promise<string[]>
  getGroupIds(suite: string): Promise<string[]>

  getHeatmapData(metric: string, suite: string, groupMetric: GroupHeatmapMetric): Promise<HeatmapData>
}

export class DataRepo implements Repo {
  constructor(private dashboardData: DashboardData) {}

  async getMetrics(suite: string): Promise<string[]> {
    return [
      ...new Set(
        this.dashboardData.policy_evals
          .filter((row) => row.suite === suite)
          .flatMap((row) => row.policy_eval_metrics.map((metric) => metric.metric))
      ),
    ].sort()
  }

  async getSuites(): Promise<string[]> {
    return [...new Set(this.dashboardData.policy_evals.map((row) => row.suite))].sort()
  }

  async getGroupIds(suite: string): Promise<string[]> {
    return [
      ...new Set(
        this.dashboardData.policy_evals
          .filter((row) => row.suite === suite)
          .flatMap((row) => row.policy_eval_metrics.map((metric) => metric.group_id))
      ),
    ].sort()
  }

  calculateValue(policyEval: PolicyEval, metric: string, groupMetric: GroupHeatmapMetric): number {
    let value = 0

    const relevantMetrics = policyEval.policy_eval_metrics.filter((m: PolicyEvalMetric) => m.metric === metric)
    if (typeof groupMetric === 'string') {
      const groupMetrics = relevantMetrics.filter(
        (m: PolicyEvalMetric) => groupMetric === '' || m.group_id === groupMetric
      )
      if (groupMetrics.length > 0) {
        const totalValue = groupMetrics.reduce((sum, m) => sum + m.sum_value, 0)
        let totalAgents = 0
        if (groupMetric === '') {
          totalAgents = Object.values(policyEval.group_num_agents).reduce((sum, num) => sum + num, 0)
        } else {
          totalAgents = policyEval.group_num_agents[groupMetric]
        }
        value = totalValue / totalAgents
      }
    } else {
      const group1Metric = relevantMetrics.find((m: PolicyEvalMetric) => m.group_id === groupMetric.group_1)
      const group2Metric = relevantMetrics.find((m: PolicyEvalMetric) => m.group_id === groupMetric.group_2)
      const group1Value = group1Metric ? group1Metric.sum_value / policyEval.group_num_agents[groupMetric.group_1] : 0
      const group2Value = group2Metric ? group2Metric.sum_value / policyEval.group_num_agents[groupMetric.group_2] : 0
      value = group1Value - group2Value
    }

    return value
  }

  async getHeatmapData(metric: string, suite: string, groupMetric: GroupHeatmapMetric): Promise<HeatmapData> {
    const suiteData = this.dashboardData.policy_evals.filter((row) => row.suite === suite)

    const getKey = (policyUri: string, evalName: string) => `${policyUri}-${evalName}`
    const policyEvalsByKey = new Map<string, PolicyEval>()
    suiteData.forEach((row) => {
      policyEvalsByKey.set(getKey(row.policy_uri, row.eval_name), row)
    })

    const evalNames = new Set<string>(suiteData.map((row) => row.eval_name))
    const policyUris = new Set<string>(this.dashboardData.policy_evals.map((row) => row.policy_uri))
    const cells = new Map<string, Map<string, HeatmapCell>>()

    for (const policyUri of policyUris) {
      for (const evalName of evalNames) {
        const key = getKey(policyUri, evalName)
        const policyEval = policyEvalsByKey.get(key)

        let cell: HeatmapCell | undefined
        if (policyEval) {
          const value = this.calculateValue(policyEval, metric, groupMetric)
          cell = {
            evalName: evalName,
            replayUrl: policyEval.replay_url,
            value: value,
          }
        } else {
          cell = {
            evalName: evalName,
            replayUrl: null,
            value: 0,
          }
        }

        let policyData = cells.get(policyUri)
        if (!policyData) {
          policyData = new Map<string, HeatmapCell>()
          cells.set(policyUri, policyData)
        }
        policyData.set(evalName, cell)
      }
    }

    const policyAverageScores = new Map<string, number>()
    cells.forEach((policyCells, policyUri) => {
      const policyAverageScore =
        Array.from(policyCells.values()).reduce((sum, cell) => sum + cell.value, 0) / evalNames.size
      policyAverageScores.set(policyUri, policyAverageScore)
    })

    const evalAverageScores = new Map<string, number>()
    const evalMaxScores = new Map<string, number>()

    cells.forEach((policyCells, _) => {
      policyCells.forEach((cell, evalName) => {
        evalAverageScores.set(evalName, (evalAverageScores.get(evalName) || 0) + cell.value)
        evalMaxScores.set(evalName, Math.max(evalMaxScores.get(evalName) || 0, cell.value))
      })
    })
    evalAverageScores.forEach((value, key) => {
      evalAverageScores.set(key, value / evalNames.size)
    })

    return { evalNames, cells, policyAverageScores, evalAverageScores, evalMaxScores }
  }
}
