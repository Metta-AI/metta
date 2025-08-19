import { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

// Register Chart.js components including Filler for area charts
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface ConfidenceData {
  num_observations: number
  running_best: number[]
  running_mean: number[]
  running_std: number[]
  convergence: {
    std: number
    slope: number
    is_converged: boolean
    estimated_runs_remaining: number | null
  }
  uncertainty: {
    average: number | null
    max: number | null
    min: number | null
    high_confidence_ratio: number | null
  }
  significance: {
    baseline_score: number
    current_best: number
    best_run_index: number
    improvement: number
    p_value: number
    is_significant: boolean
  }
  timestamps: number[]
  error?: string
  message?: string
}

interface ModelConfidenceProps {
  sweepName: string
}

export function ModelConfidence({ sweepName }: ModelConfidenceProps) {
  const [data, setData] = useState<ConfidenceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchConfidenceData()
    const interval = setInterval(fetchConfidenceData, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [sweepName])

  const fetchConfidenceData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/sweeps/${sweepName}/confidence`)
      const result = await response.json()
      
      if (result.error) {
        setError(result.message || 'Error loading confidence metrics')
        setData(null)
      } else {
        setData(result)
        setError(null)
      }
    } catch (err) {
      console.error('Failed to fetch confidence data:', err)
      setError('Failed to load confidence metrics')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          Model Confidence Dashboard
        </h3>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div className="spinner" />
          <div style={{ marginTop: '16px', color: '#6c757d' }}>
            Loading confidence metrics...
          </div>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          Model Confidence Dashboard
        </h3>
        <div className="alert alert-warning">
          {error || 'No confidence data available'}
        </div>
      </div>
    )
  }

  // Prepare convergence plot data
  const convergenceData = {
    labels: data.timestamps,
    datasets: [
      {
        label: 'Best Score',
        data: data.running_best,
        borderColor: 'rgb(40, 167, 69)',
        backgroundColor: 'rgba(40, 167, 69, 0.1)',
        borderWidth: 2,
        tension: 0.1,
        pointRadius: 2,
      },
      {
        label: 'Mean ± Std',
        data: data.running_mean,
        borderColor: 'rgb(0, 123, 255)',
        backgroundColor: 'rgba(0, 123, 255, 0.1)',
        borderWidth: 1,
        borderDash: [5, 5],
        tension: 0.1,
        pointRadius: 0,
        fill: false,
      },
      {
        label: 'Upper Bound (Mean + Std)',
        data: data.running_mean.map((m, i) => m + data.running_std[i]),
        borderColor: 'rgba(0, 123, 255, 0.3)',
        backgroundColor: 'rgba(0, 123, 255, 0.05)',
        borderWidth: 1,
        tension: 0.1,
        pointRadius: 0,
        fill: '+1',
      },
      {
        label: 'Lower Bound (Mean - Std)',
        data: data.running_mean.map((m, i) => m - data.running_std[i]),
        borderColor: 'rgba(0, 123, 255, 0.3)',
        backgroundColor: 'rgba(0, 123, 255, 0.05)',
        borderWidth: 1,
        tension: 0.1,
        pointRadius: 0,
        fill: '-1',
      },
    ],
  }

  const convergenceOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          filter: (item: any) => !item.text.includes('Bound'),
        },
      },
      title: {
        display: true,
        text: 'Convergence Plot with Confidence Bands',
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Observation Number',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Score',
        },
      },
    },
  }

  // Format percentage
  const formatPercent = (value: number | null) => {
    if (value === null) return 'N/A'
    return `${(value * 100).toFixed(1)}%`
  }

  // Format p-value
  const formatPValue = (value: number) => {
    if (value < 0.001) return '< 0.001'
    return value.toFixed(3)
  }

  return (
    <div className="card">
      <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>
        Model Confidence Dashboard
      </h3>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-4" style={{ marginBottom: '24px' }}>
        {/* Convergence Status */}
        <div style={{ 
          padding: '16px',
          backgroundColor: data.convergence.is_converged ? '#d4edda' : '#fff3cd',
          borderRadius: '6px',
          border: `1px solid ${data.convergence.is_converged ? '#c3e6cb' : '#ffeeba'}`
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
            Convergence Status
          </div>
          <div style={{ 
            fontSize: '18px', 
            fontWeight: 600,
            color: data.convergence.is_converged ? '#155724' : '#856404'
          }}>
            {data.convergence.is_converged ? '✓ Converged' : '⟳ Optimizing'}
          </div>
          <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
            Std: {data.convergence.std.toFixed(4)}
          </div>
        </div>

        {/* Statistical Significance */}
        <div style={{ 
          padding: '16px',
          backgroundColor: data.significance.is_significant ? '#d4edda' : '#f8d7da',
          borderRadius: '6px',
          border: `1px solid ${data.significance.is_significant ? '#c3e6cb' : '#f5c6cb'}`
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
            Statistical Significance
          </div>
          <div style={{ 
            fontSize: '18px', 
            fontWeight: 600,
            color: data.significance.is_significant ? '#155724' : '#721c24'
          }}>
            {data.significance.is_significant ? '✓ Significant' : '✗ Not Significant'}
          </div>
          <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
            p-value: {formatPValue(data.significance.p_value)}
          </div>
        </div>

        {/* Model Uncertainty */}
        <div style={{ 
          padding: '16px',
          backgroundColor: '#e7f3ff',
          borderRadius: '6px',
          border: '1px solid #b3d9ff'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
            Average Uncertainty
          </div>
          <div style={{ fontSize: '18px', fontWeight: 600, color: '#004085' }}>
            {data.uncertainty.average !== null ? data.uncertainty.average.toFixed(3) : 'N/A'}
          </div>
          <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
            Range: [{data.uncertainty.min?.toFixed(3) || 'N/A'}, {data.uncertainty.max?.toFixed(3) || 'N/A'}]
          </div>
        </div>

        {/* High Confidence Regions */}
        <div style={{ 
          padding: '16px',
          backgroundColor: '#f0f0f0',
          borderRadius: '6px',
          border: '1px solid #d0d0d0'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
            High Confidence Space
          </div>
          <div style={{ fontSize: '18px', fontWeight: 600, color: '#495057' }}>
            {formatPercent(data.uncertainty.high_confidence_ratio)}
          </div>
          <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
            of parameter space
          </div>
        </div>
      </div>

      {/* Progress Summary */}
      <div style={{ 
        padding: '16px',
        backgroundColor: '#f8f9fa',
        borderRadius: '6px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
          <div>
            <div style={{ fontSize: '12px', color: '#6c757d', marginBottom: '4px' }}>
              Observations
            </div>
            <div style={{ fontSize: '20px', fontWeight: 600 }}>
              {data.num_observations}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '12px', color: '#6c757d', marginBottom: '4px' }}>
              Improvement
            </div>
            <div style={{ fontSize: '20px', fontWeight: 600, color: '#28a745' }}>
              +{data.significance.improvement.toFixed(4)}
            </div>
            <div style={{ fontSize: '11px', color: '#6c757d' }}>
              from {data.significance.baseline_score.toFixed(4)} → {data.significance.current_best.toFixed(4)}
              {data.significance.best_run_index > 0 && ` (run #${data.significance.best_run_index + 1})`}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '12px', color: '#6c757d', marginBottom: '4px' }}>
              Est. Runs to Convergence
            </div>
            <div style={{ fontSize: '20px', fontWeight: 600 }}>
              {data.convergence.estimated_runs_remaining !== null 
                ? data.convergence.estimated_runs_remaining 
                : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Convergence Plot */}
      <div style={{ height: '300px', marginBottom: '20px' }}>
        <Line data={convergenceData} options={convergenceOptions} />
      </div>

      {/* Convergence Metrics Details */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
        fontSize: '13px'
      }}>
        <div style={{ padding: '12px', backgroundColor: '#f8f9fa', borderRadius: '6px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>
            Convergence Metrics
          </h4>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#6c757d' }}>Recent Std Dev:</span>
            <span style={{ fontWeight: 500 }}>{data.convergence.std.toFixed(6)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#6c757d' }}>Improvement Slope:</span>
            <span style={{ fontWeight: 500 }}>{data.convergence.slope.toFixed(6)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#6c757d' }}>Status:</span>
            <span style={{ 
              fontWeight: 500,
              color: data.convergence.is_converged ? '#28a745' : '#ffc107'
            }}>
              {data.convergence.is_converged ? 'Converged' : 'Still Improving'}
            </span>
          </div>
        </div>

        <div style={{ padding: '12px', backgroundColor: '#f8f9fa', borderRadius: '6px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>
            Confidence Summary
          </h4>
          <div style={{ marginBottom: '8px' }}>
            {data.uncertainty.average !== null && data.uncertainty.average < 0.1 ? (
              <div style={{ color: '#28a745' }}>
                ✓ Low uncertainty - Model is confident
              </div>
            ) : data.uncertainty.average !== null && data.uncertainty.average < 0.5 ? (
              <div style={{ color: '#ffc107' }}>
                ⚠ Moderate uncertainty - More exploration needed
              </div>
            ) : (
              <div style={{ color: '#dc3545' }}>
                ✗ High uncertainty - Significant exploration required
              </div>
            )}
          </div>
          <div>
            {data.convergence.estimated_runs_remaining !== null && data.convergence.estimated_runs_remaining < 20 ? (
              <div style={{ color: '#17a2b8' }}>
                ℹ Near convergence (~{data.convergence.estimated_runs_remaining} runs)
              </div>
            ) : data.convergence.is_converged ? (
              <div style={{ color: '#28a745' }}>
                ✓ Optimization complete
              </div>
            ) : (
              <div style={{ color: '#6c757d' }}>
                ⟳ Continue optimization
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}