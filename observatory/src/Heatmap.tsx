import { useState } from 'react'
import Plot from 'react-plotly.js'
import { HeatmapData } from './repo'

interface HeatmapProps {
  data: HeatmapData
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

const getShortName = (evalName: string) => {
  if (evalName === 'Overall') return evalName
  return evalName.split('/').pop() || evalName
}

const wandb_url = (policyName: string) => {
  const entity = 'metta-research'
  const project = 'metta'
  let policyKey = policyName
  if (policyName.includes(':v')) {
    policyKey = policyName.split(':v')[0]
  }
  return `https://wandb.ai/${entity}/${project}/runs/${policyKey}`
}

export function Heatmap({
  data,
  selectedMetric,
  setSelectedCell,
  openReplayUrl,
  numPoliciesToShow = 20,
}: HeatmapProps) {
  const [lastHoveredCell, setLastHoveredCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)

  // Convert to heatmap format
  const policies = Object.keys(data.cells)
  const shortNameToEvalName = new Map<string, string>()
  data.evalNames.forEach((evalName) => {
    shortNameToEvalName.set(getShortName(evalName), evalName)
  })
  const sortedShortNames = [...shortNameToEvalName.keys()].sort((a, b) => a.localeCompare(b))

  const xLabels = ['overall', ...sortedShortNames]

  // Iterate over the policyEvalMap, and for each policy compute the average value of the evals
  const sortedPolicies = policies.sort((a, b) => data.policyAverageScores[a] - data.policyAverageScores[b])
  // take last 20 of sorted policies
  const y_labels = sortedPolicies.slice(-numPoliciesToShow)

  const z = y_labels.map((policy) => [
    data.policyAverageScores[policy],
    ...sortedShortNames.map((shortName) => {
      const evalName = shortNameToEvalName.get(shortName)!
      const cell = data.cells[policy]?.[evalName]
      if (!cell) {
        return 0
      }
      return cell.value
    }),
  ])

  const y_label_texts = y_labels.map((policy) => {
    return `<a href="${wandb_url(policy)}" target="_blank">${policy}</a>`
  })

  const onHover = (event: any) => {
    if (!event.points?.[0]) return

    const shortName = event.points[0].x
    const policyUri = event.points[0].y

    const evalName = shortNameToEvalName.get(shortName)!

    setLastHoveredCell({ policyUri, evalName })
    if (!(shortName === 'overall')) {
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
