import { SweepData } from '../types'

interface SweepMetricsProps {
  data: SweepData
}

export function SweepMetrics({ data }: SweepMetricsProps) {
  return (
    <div className="grid grid-cols-4" style={{ marginBottom: '20px' }}>
      <div className="card" style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '24px', fontWeight: 600, color: '#007bff', marginBottom: '8px' }}>
          {data.totalRuns}
        </div>
        <div style={{ fontSize: '14px', color: '#6c757d' }}>Total Runs</div>
      </div>

      <div className="card" style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '24px', fontWeight: 600, color: '#28a745', marginBottom: '8px' }}>
          {data.bestScore.toFixed(4)}
        </div>
        <div style={{ fontSize: '14px', color: '#6c757d' }}>Best Score</div>
      </div>

      <div className="card" style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '24px', fontWeight: 600, color: '#ffc107', marginBottom: '8px' }}>
          ${data.totalCost.toFixed(2)}
        </div>
        <div style={{ fontSize: '14px', color: '#6c757d' }}>Total Cost</div>
      </div>

      <div className="card" style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '24px', fontWeight: 600, color: '#17a2b8', marginBottom: '8px' }}>
          {(data.avgRuntime / 60).toFixed(1)}m
        </div>
        <div style={{ fontSize: '14px', color: '#6c757d' }}>Avg Runtime</div>
      </div>
    </div>
  )
}