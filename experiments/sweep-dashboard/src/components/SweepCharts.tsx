import { useEffect, useRef, useMemo } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js'
import { Line, Scatter, Bar } from 'react-chartjs-2'
import { SweepData } from '../types'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

interface SweepChartsProps {
  data: SweepData
}

export function SweepCharts({ data }: SweepChartsProps) {
  // Sort runs by timestamp for timeline chart
  const sortedRuns = [...data.runs].sort((a, b) => 
    new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )

  // Calculate parameter correlations
  const parameterCorrelations = useMemo(() => {
    const correlations: { parameter: string; correlation: number }[] = []
    
    // Get numeric parameters only
    const paramKeys = data.parameters.filter(param => {
      const values = data.runs.map(r => r.parameters[param]).filter(v => v !== undefined)
      return values.length > 0 && typeof values[0] === 'number'
    })

    paramKeys.forEach(param => {
      const values = data.runs.map(r => ({
        x: r.parameters[param] as number,
        y: r.score
      })).filter(v => v.x !== undefined)

      if (values.length >= 2) {
        // Calculate Pearson correlation
        const n = values.length
        const sumX = values.reduce((acc, v) => acc + v.x, 0)
        const sumY = values.reduce((acc, v) => acc + v.y, 0)
        const sumXY = values.reduce((acc, v) => acc + v.x * v.y, 0)
        const sumX2 = values.reduce((acc, v) => acc + v.x * v.x, 0)
        const sumY2 = values.reduce((acc, v) => acc + v.y * v.y, 0)

        const correlation = (n * sumXY - sumX * sumY) / 
          Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

        if (!isNaN(correlation)) {
          correlations.push({
            parameter: param.replace('trainer.', ''),
            correlation
          })
        }
      }
    })

    return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
  }, [data])

  // Calculate Pareto frontier
  const paretoFrontier = useMemo(() => {
    const sorted = [...data.runs].sort((a, b) => a.cost - b.cost)
    const frontier: typeof data.runs = []
    let maxScore = -Infinity

    sorted.forEach(run => {
      if (run.score >= maxScore) {
        frontier.push(run)
        maxScore = run.score
      }
    })

    return frontier
  }, [data.runs])

  // Cost vs Score scatter plot data
  const costVsScoreData = {
    datasets: [{
      label: 'Runs',
      data: data.runs.map(run => ({
        x: run.cost,
        y: run.score
      })),
      backgroundColor: 'rgba(0, 123, 255, 0.5)',
      borderColor: 'rgba(0, 123, 255, 1)',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
    }, {
      label: 'Best Score',
      data: [{
        x: data.runs.find(r => r.score === data.bestScore)?.cost || 0,
        y: data.bestScore
      }],
      backgroundColor: 'red',
      borderColor: 'darkred',
      borderWidth: 2,
      pointRadius: 10,
      pointStyle: 'star',
      showLine: false
    }]
  }

  const costVsScoreOptions: ChartOptions<'scatter'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Cost vs Score Analysis'
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw as { x: number, y: number }
            return `Cost: $${point.x.toFixed(2)}, Score: ${point.y.toFixed(4)}`
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Cost ($)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Score'
        }
      }
    }
  }

  // Timeline chart data with moving average
  const windowSize = Math.min(5, Math.floor(sortedRuns.length / 3))
  const movingAverage = sortedRuns.map((_, i) => {
    const start = Math.max(0, i - Math.floor(windowSize / 2))
    const end = Math.min(sortedRuns.length, start + windowSize)
    const window = sortedRuns.slice(start, end)
    return window.reduce((sum, r) => sum + r.score, 0) / window.length
  })

  const timelineData = {
    labels: sortedRuns.map((_, i) => `Run ${i + 1}`),
    datasets: [{
      label: 'Score Over Time',
      data: sortedRuns.map(run => run.score),
      borderColor: 'rgb(40, 167, 69)',
      backgroundColor: 'rgba(40, 167, 69, 0.1)',
      tension: 0.1
    }, {
      label: `Moving Avg (window=${windowSize})`,
      data: movingAverage,
      borderColor: 'red',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3
    }]
  }

  const timelineOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Score Progression Over Time'
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Score'
        }
      }
    }
  }

  // Parameter Importance bar chart
  const paramImportanceData = {
    labels: parameterCorrelations.slice(0, 15).map(p => p.parameter),
    datasets: [{
      label: 'Correlation with Score',
      data: parameterCorrelations.slice(0, 15).map(p => p.correlation),
      backgroundColor: parameterCorrelations.slice(0, 15).map(p => 
        p.correlation > 0 ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)'
      ),
      borderColor: parameterCorrelations.slice(0, 15).map(p => 
        p.correlation > 0 ? 'rgb(40, 167, 69)' : 'rgb(220, 53, 69)'
      ),
      borderWidth: 1
    }]
  }

  const paramImportanceOptions: ChartOptions<'bar'> = {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Top 15 Parameter Correlations with Score'
      },
      tooltip: {
        callbacks: {
          label: (context) => `Correlation: ${(context.raw as number).toFixed(3)}`
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Correlation'
        },
        min: -1,
        max: 1
      }
    }
  }

  // Pareto Frontier plot
  const paretoData = {
    datasets: [{
      label: 'All Runs',
      data: data.runs.map(run => ({
        x: run.cost,
        y: run.score
      })),
      backgroundColor: 'rgba(200, 200, 200, 0.5)',
      borderColor: 'rgba(200, 200, 200, 1)',
      borderWidth: 1,
      pointRadius: 4,
    }, {
      label: 'Pareto Frontier',
      data: paretoFrontier.map(run => ({
        x: run.cost,
        y: run.score
      })),
      backgroundColor: 'rgba(255, 0, 0, 0.7)',
      borderColor: 'red',
      borderWidth: 2,
      pointRadius: 8,
      showLine: true,
      tension: 0
    }]
  }

  const paretoOptions: ChartOptions<'scatter'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Efficiency Frontier (Pareto Optimal Runs)'
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw as { x: number, y: number }
            const label = context.dataset.label === 'Pareto Frontier' ? 'Efficient - ' : ''
            return `${label}Cost: $${point.x.toFixed(2)}, Score: ${point.y.toFixed(4)}`
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Cost ($)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Score'
        }
      }
    }
  }

  // Score distribution histogram
  const scoreHistogram = {
    labels: Array.from({ length: 10 }, (_, i) => {
      const min = Math.min(...data.runs.map(r => r.score))
      const max = Math.max(...data.runs.map(r => r.score))
      const range = max - min
      const binSize = range / 10
      const start = min + i * binSize
      return start.toFixed(3)
    }),
    datasets: [{
      label: 'Score Distribution',
      data: Array.from({ length: 10 }, (_, i) => {
        const min = Math.min(...data.runs.map(r => r.score))
        const max = Math.max(...data.runs.map(r => r.score))
        const range = max - min
        const binSize = range / 10
        const start = min + i * binSize
        const end = start + binSize
        return data.runs.filter(r => r.score >= start && r.score < end).length
      }),
      backgroundColor: 'rgba(40, 167, 69, 0.5)',
      borderColor: 'rgba(40, 167, 69, 1)',
      borderWidth: 1
    }]
  }

  const histogramOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Score Distribution'
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Count'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Score'
        }
      }
    }
  }

  // Cost distribution histogram
  const costHistogram = {
    labels: Array.from({ length: 10 }, (_, i) => {
      const min = Math.min(...data.runs.map(r => r.cost))
      const max = Math.max(...data.runs.map(r => r.cost))
      const range = max - min
      const binSize = range / 10
      const start = min + i * binSize
      return `$${start.toFixed(2)}`
    }),
    datasets: [{
      label: 'Cost Distribution',
      data: Array.from({ length: 10 }, (_, i) => {
        const min = Math.min(...data.runs.map(r => r.cost))
        const max = Math.max(...data.runs.map(r => r.cost))
        const range = max - min
        const binSize = range / 10
        const start = min + i * binSize
        const end = start + binSize
        return data.runs.filter(r => r.cost >= start && r.cost < end).length
      }),
      backgroundColor: 'rgba(255, 193, 7, 0.5)',
      borderColor: 'rgba(255, 193, 7, 1)',
      borderWidth: 1
    }]
  }

  // Parameter Correlation Scatter Plots
  const topParams = parameterCorrelations.slice(0, 6).map(p => p.parameter)

  return (
    <div>
      {/* Main visualizations - same as Plotly dashboard */}
      <div className="grid grid-cols-2">
        <div className="card">
          <Scatter data={costVsScoreData} options={costVsScoreOptions} />
        </div>
        <div className="card">
          <Bar data={paramImportanceData} options={paramImportanceOptions} />
        </div>
      </div>

      <div className="grid grid-cols-2">
        <div className="card">
          <Line data={timelineData} options={timelineOptions} />
        </div>
        <div className="card">
          <Scatter data={paretoData} options={paretoOptions} />
        </div>
      </div>

      {/* Distribution plots */}
      <div className="grid grid-cols-2">
        <div className="card">
          <Bar data={scoreHistogram} options={histogramOptions} />
        </div>
        <div className="card">
          <Bar data={costHistogram} options={histogramOptions} />
        </div>
      </div>

      {/* Parameter Correlation Plots - Grid of scatter plots */}
      <div className="card">
        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '20px' }}>
          Parameter Correlations with Score
        </h3>
        <div className="grid grid-cols-3" style={{ gap: '20px' }}>
          {topParams.map(param => {
            const fullParam = data.parameters.find(p => p.includes(param)) || param
            const paramData = data.runs
              .map(run => ({
                x: run.parameters[fullParam] as number,
                y: run.score
              }))
              .filter(v => v.x !== undefined)

            // Calculate trend line
            const n = paramData.length
            if (n < 2) return null

            const sumX = paramData.reduce((acc, v) => acc + v.x, 0)
            const sumY = paramData.reduce((acc, v) => acc + v.y, 0)
            const sumXY = paramData.reduce((acc, v) => acc + v.x * v.y, 0)
            const sumX2 = paramData.reduce((acc, v) => acc + v.x * v.x, 0)

            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
            const intercept = (sumY - slope * sumX) / n

            const minX = Math.min(...paramData.map(d => d.x))
            const maxX = Math.max(...paramData.map(d => d.x))

            const correlation = parameterCorrelations.find(c => c.parameter === param)?.correlation || 0

            const scatterData = {
              datasets: [{
                label: 'Data Points',
                data: paramData,
                backgroundColor: 'rgba(0, 123, 255, 0.5)',
                borderColor: 'rgba(0, 123, 255, 1)',
                borderWidth: 1,
                pointRadius: 4
              }, {
                label: 'Trend',
                data: [
                  { x: minX, y: slope * minX + intercept },
                  { x: maxX, y: slope * maxX + intercept }
                ],
                type: 'line' as const,
                borderColor: 'red',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0
              }]
            }

            const options: ChartOptions<'scatter'> = {
              responsive: true,
              plugins: {
                title: {
                  display: true,
                  text: `${param} (r=${correlation.toFixed(3)})`
                },
                legend: {
                  display: false
                }
              },
              scales: {
                x: {
                  title: {
                    display: true,
                    text: param,
                    font: {
                      size: 10
                    }
                  }
                },
                y: {
                  title: {
                    display: true,
                    text: 'Score',
                    font: {
                      size: 10
                    }
                  }
                }
              }
            }

            return (
              <div key={param}>
                <Scatter data={scatterData} options={options} />
              </div>
            )
          })}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>
          Summary Statistics
        </h3>
        <div className="grid grid-cols-2" style={{ gap: '20px' }}>
          <div>
            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>Score Statistics</h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li>Mean: {(data.runs.reduce((sum, r) => sum + r.score, 0) / data.runs.length).toFixed(4)}</li>
              <li>Median: {data.runs.sort((a, b) => a.score - b.score)[Math.floor(data.runs.length / 2)]?.score.toFixed(4)}</li>
              <li>Min: {Math.min(...data.runs.map(r => r.score)).toFixed(4)}</li>
              <li>Max: {Math.max(...data.runs.map(r => r.score)).toFixed(4)}</li>
              <li>Std Dev: {Math.sqrt(
                data.runs.reduce((sum, r) => sum + Math.pow(r.score - data.runs.reduce((s, r2) => s + r2.score, 0) / data.runs.length, 2), 0) / data.runs.length
              ).toFixed(4)}</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px' }}>Cost Statistics</h4>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li>Total: ${data.totalCost.toFixed(2)}</li>
              <li>Mean: ${(data.totalCost / data.runs.length).toFixed(2)}</li>
              <li>Median: ${data.runs.sort((a, b) => a.cost - b.cost)[Math.floor(data.runs.length / 2)]?.cost.toFixed(2)}</li>
              <li>Min: ${Math.min(...data.runs.map(r => r.cost)).toFixed(2)}</li>
              <li>Max: ${Math.max(...data.runs.map(r => r.cost)).toFixed(2)}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}