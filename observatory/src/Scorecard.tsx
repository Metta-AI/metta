import { useState } from 'react'
import Plot from 'react-plotly.js'
import { ScorecardData } from './repo'
import { METTA_WANDB_ENTITY, METTA_WANDB_PROJECT } from './constants'
import { getShortName, groupEvalNamesByCategory, reconstructEvalName, OVERALL_EVAL_NAME } from './utils/evalNameUtils'

interface ScorecardProps {
  data: ScorecardData
  selectedMetric: string
  setSelectedCell: (cell: { policyUri: string; evalName: string }) => void
  openReplayUrl: (policyUri: string, evalName: string) => void
  numPoliciesToShow: number
}

// CSS for tabs
const SUITE_TABS_CSS = `
.suite-tabs {
  display: flex;
  gap: 2px;
  padding: 4px;
  border-radius: 8px;
  margin-bottom: 20px;
  overflow-x: auto;
  max-width: 1000px;
  margin: 0 auto 20px auto;
}

.suite-tab {
  padding: 8px 16px;
  border: none;
  background: #fff;
  cursor: pointer;
  font-size: 14px;
  color: #666;
  border-radius: 6px;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.suite-tab:hover {
  background: #f8f8f8;
  color: #333;
}

.suite-tab.active {
  background: #007bff;
  color: #fff;
  font-weight: 500;
}

.suite-tab:first-child {
  margin-left: 0;
}

.suite-tab:last-child {
  margin-right: 0;
}
`

const wandb_url = (policyName: string) => {
  const entity = METTA_WANDB_ENTITY
  const project = METTA_WANDB_PROJECT
  let policyKey = policyName
  if (policyName.includes(':v')) {
    policyKey = policyName.split(':v')[0]
  }
  return `https://wandb.ai/${entity}/${project}/runs/${policyKey}`
}

export function Scorecard({
  data,
  selectedMetric,
  setSelectedCell,
  openReplayUrl,
  numPoliciesToShow = 20,
}: ScorecardProps) {
  const [lastHoveredCell, setLastHoveredCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)

  // Convert to scorecard format
  const policies = Object.keys(data.cells)

  // Group eval names by category for better organization
  const evalsByCategory = groupEvalNamesByCategory(data.evalNames)

  // Build x-labels: overall, then grouped by category
  const xLabels = [OVERALL_EVAL_NAME]
  const shortNameToEvalName = new Map<string, string>()
  shortNameToEvalName.set(OVERALL_EVAL_NAME, OVERALL_EVAL_NAME)

  // Helper to iterate over categories and environments safely
  const forEachCategoryEnv = (fn: (category: string, envName: string, fullEvalName: string) => void) => {
    const sortedCategories = Array.from(evalsByCategory.keys()).sort()
    for (const category of sortedCategories) {
      const envNames = evalsByCategory.get(category)
      if (!envNames) {
        throw new Error(`No environment names found for category: ${category}`)
      }

      const sortedEnvNames = envNames.sort()
      for (const envName of sortedEnvNames) {
        const fullEvalName = reconstructEvalName(category, envName)
        fn(category, envName, fullEvalName)
      }
    }
  }

  // Build x-labels and mapping
  forEachCategoryEnv((_category, _envName, fullEvalName) => {
    const shortName = getShortName(fullEvalName)
    xLabels.push(shortName)
    shortNameToEvalName.set(shortName, fullEvalName)
  })

  // Sort policies by average score (best at bottom for better visibility)
  const sortedPolicies = policies.sort((a, b) => data.policyAverageScores[a] - data.policyAverageScores[b])
  // Take the specified number of top policies (from the end of sorted list)
  const y_labels = sortedPolicies.slice(-numPoliciesToShow)

  const z = y_labels.map((policy) => {
    const row = [data.policyAverageScores[policy]] // Overall score first

    // Add scores for each evaluation in order
    forEachCategoryEnv((_, __, fullEvalName) => {
      const cell = data.cells[policy]?.[fullEvalName]
      row.push(cell ? cell.value : 0)
    })

    return row
  })

  const y_label_texts = y_labels.map((policy) => {
    return `<a href="${wandb_url(policy)}" target="_blank">${policy}</a>`
  })

  const onHover = (event: any) => {
    if (!event.points?.[0]) return

    const shortName = event.points[0].x
    const policyUri = event.points[0].y

    const evalName = shortNameToEvalName.get(shortName)
    if (!evalName) {
      throw new Error(`No eval name found for short name: ${shortName}`)
    }

    setLastHoveredCell({ policyUri, evalName })
    if (!(shortName === OVERALL_EVAL_NAME)) {
      setSelectedCell({ policyUri, evalName })
    }
  }

  const onDoubleClick = () => {
    if (lastHoveredCell) {
      openReplayUrl(lastHoveredCell.policyUri, lastHoveredCell.evalName)
    }
  }

  const plotData: Plotly.Data = {
    z,
    x: xLabels,
    y: y_labels,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: {
        text: selectedMetric,
      },
    },
  }

  return (
    <>
      <style>{SUITE_TABS_CSS}</style>
      <Plot
        data={[plotData]}
        layout={{
          title: {
            text: `Policy Evaluation Report: ${selectedMetric}`,
            font: {
              size: 24,
            },
          },
          height: 600,
          width: 1000,
          margin: { t: 50, b: 150, l: 200, r: 50 },
          xaxis: {
            tickangle: -45,
          },
          yaxis: {
            tickangle: 0,
            automargin: true,
            ticktext: y_label_texts,
            tickvals: Array.from({ length: y_labels.length }, (_, i) => i),
            tickmode: 'array',
          },
        }}
        style={{
          margin: '0 auto',
          display: 'block',
        }}
        onHover={onHover}
        onDoubleClick={onDoubleClick}
      />
    </>
  )
}
