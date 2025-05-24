import Plot from 'react-plotly.js';
import { PolicyEvalMetric } from './data_loader';

interface HeatmapProps {
  matrix: PolicyEvalMetric[];
  selectedMetric: string;
  onHover: (event: any) => void;
  onDoubleClick: () => void;
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
`;

const getShortName = (evalName: string) => {
  if (evalName === 'Overall') return evalName;
  return evalName.split('/').pop() || evalName;
};

const wandb_url = (policyName: string) => {
  const entity = "metta-research"
  const project = "metta"
  let policyKey = policyName
  if (policyName.includes(":v")) {
    policyKey = policyName.split(":v")[0]
  }
  return `https://wandb.ai/${entity}/${project}/runs/${policyKey}`
};

export function Heatmap({ matrix, selectedMetric, onHover, onDoubleClick }: HeatmapProps) {
  const policyEvalMap = new Map<string, Map<string, number>>();
  for (const row of matrix) {
    if (!policyEvalMap.has(row.policy_uri)) {
      policyEvalMap.set(row.policy_uri, new Map())
    }
    policyEvalMap.get(row.policy_uri)?.set(getShortName(row.eval_name), row.value)
  }

  // Convert to heatmap format
  const policies = [...new Set(matrix.map(r => r.policy_uri))]
  const envs = [...new Set(matrix.map(r => r.eval_name))]
  const sortedShortNames = envs.map(getShortName).sort((a, b) => a.localeCompare(b));

  const sortedShortNamesWithOverall = ["overall", ...sortedShortNames];

  // Iterate over the policyEvalMap, and for each policy compute the average value of the evals
  policyEvalMap.forEach((evalMap) => {
    const overallValue = Array.from(evalMap.values()).reduce((sum, value) => sum + value, 0) / envs.length;
    evalMap.set("overall", overallValue);
  });
  const sortedPolicies = policies.sort((a, b) => policyEvalMap.get(a)!.get("overall")! - policyEvalMap.get(b)!.get("overall")!);
  // take last 20 of sorted policies
  const y_labels = sortedPolicies.slice(-20)

  const z = y_labels.map(policy => 
    sortedShortNamesWithOverall.map(shortName => policyEvalMap.get(policy)!.get(shortName) || 0)
  )

  const y_label_texts = y_labels.map(policy => {
    return `<a href="${wandb_url(policy)}" target="_blank">${policy}</a>`
  })

  const data: Plotly.Data = {
    z,
    x: sortedShortNamesWithOverall,
    y: y_labels,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: {
        text: selectedMetric
      }
    }
  }

  return (
    <>
      <style>{SUITE_TABS_CSS}</style>
      <Plot
        data={[data]}
        layout={{
          title: {
            text: `Policy Evaluation Report: ${selectedMetric}`,
            font: {
              size: 24
            }
          },
          height: 600,
          width: 1000,
          margin: { t: 50, b: 150, l: 200, r: 50 },
          xaxis: {
            tickangle: -45
          },
          yaxis: {
            tickangle: 0,
            automargin: true,
            ticktext: y_label_texts,
            tickvals: Array.from({ length: y_labels.length }, (_, i) => i),
            tickmode: 'array'
          }
        }}
        style={{
          margin: '0 auto',
          display: 'block'
        }}
        onHover={onHover}
        onDoubleClick={onDoubleClick}
      />
    </>
  );
} 