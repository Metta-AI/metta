import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { HeatmapData, EpochHeatmapData, Repo } from './repo'

const POLICY_VIEW_CSS = `
.policy-view-container {
  padding: 20px;
  background: #f8f9fa;
  min-height: calc(100vh - 60px);
}

.policy-view-content {
  max-width: 1400px;
  margin: 0 auto;
  background: #fff;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.policy-header {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.breadcrumb {
  margin-bottom: 15px;
  font-size: 14px;
}

.breadcrumb a {
  color: #007bff;
  text-decoration: none;
}

.breadcrumb a:hover {
  text-decoration: underline;
}

.breadcrumb-separator {
  margin: 0 8px;
  color: #666;
}

.policy-title {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 24px;
  font-weight: 600;
}

.policy-meta {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #666;
}

.policy-meta-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.policy-status {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
}

.policy-status.active {
  background: #e8f5e8;
  color: #2e7d32;
}

.policy-status.inactive {
  background: #ffebee;
  color: #c62828;
}

.metrics-section {
  margin-bottom: 40px;
}

.metrics-section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #333;
}

.training-reward-chart {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.sample-runs-section {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.evaluation-scores-section {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.tabs-container {
  margin-bottom: 20px;
}

.tabs {
  display: flex;
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 20px;
}

.tab {
  padding: 10px 20px;
  cursor: pointer;
  border: none;
  background: none;
  color: #666;
  font-size: 14px;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.tab:hover {
  color: #333;
  background: #f8f9fa;
}

.tab.active {
  color: #007bff;
  border-bottom-color: #007bff;
}

.grid-container {
  position: relative;
  margin-bottom: 20px;
}

.runs-grid {
  display: grid;
  grid-template-columns: repeat(15, 1fr);
  gap: 2px;
  margin-bottom: 20px;
}

.grid-cell {
  width: 30px;
  height: 30px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.grid-cell:hover {
  transform: scale(1.1);
  z-index: 10;
}

.completion-indicators {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-left: 10px;
}

.completion-indicator {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #ccc;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: #666;
}

.evaluation-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  max-width: 100%;
  overflow-x: auto;
}

.evaluation-header {
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 500;
  color: #333;
  background: #f8f9fa;
  border-right: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
}

.evaluation-row-header {
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 500;
  color: #333;
  background: #f8f9fa;
  border-right: 1px solid #e0e0e0;
}

.evaluation-cell {
  min-height: 30px;
  cursor: pointer;
  transition: all 0.2s ease;
  border-right: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
  background: #fff;
  text-align: center;
  vertical-align: middle;
}

.evaluation-cell:hover {
  background: #f0f8ff;
  z-index: 10;
}

.table-container {
  overflow-x: auto;
  margin-bottom: 20px;
}

.policy-evaluations-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  max-width: 100%;
  overflow-x: auto;
}

.policy-evaluations-table th {
  padding: 12px 8px;
  font-size: 12px;
  font-weight: 600;
  color: #333;
  background: #f8f9fa;
  border-right: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
  text-align: center;
  white-space: nowrap;
}

.policy-evaluations-table td {
  padding: 8px;
  font-size: 11px;
  border-right: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
  text-align: center;
  vertical-align: middle;
}

.eval-name-cell {
  font-weight: 500;
  background: #f8f9fa;
  text-align: left;
  white-space: nowrap;
  min-width: 120px;
}

.value-cell {
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  color: #333;
  min-width: 60px;
  max-width: 80px;
}

.value-cell:hover {
  transform: scale(1.05);
  z-index: 10;
  position: relative;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.completion-cell {
  background: #f8f9fa;
  text-align: center;
}

.completion-indicator {
  display: inline-block;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #007bff;
  color: white;
  font-size: 10px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
}

.map-popup {
  position: absolute;
  background: white;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  max-width: 300px;
}

.map-popup-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.map-popup-title {
  font-weight: 600;
  font-size: 14px;
}

.map-popup-upload {
  cursor: pointer;
  color: #007bff;
  font-size: 16px;
}

.map-preview {
  width: 100%;
  height: 150px;
  background: #f0f0f0;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #666;
  font-size: 12px;
}

.additional-stats-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.stat-chart {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 20px;
}

.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #666;
}

.error-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #d32f2f;
}
`

interface PolicyViewProps {
    repo: Repo
}

type MetricTab = string

export function PolicyView({ repo }: PolicyViewProps) {
    const { policyName } = useParams<{ policyName: string }>()
    const [policy, setPolicy] = useState<any | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedTab, setSelectedTab] = useState<MetricTab>('')
    const [selectedEvalTab, setSelectedEvalTab] = useState<MetricTab>('')
    const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null)
    const [mapPopup, setMapPopup] = useState<{ x: number; y: number; visible: boolean } | null>(null)

    // Heatmap data states
    const [sampleRunsHeatmap, setSampleRunsHeatmap] = useState<EpochHeatmapData | null>(null)
    const [evaluationHeatmap, setEvaluationHeatmap] = useState<HeatmapData | null>(null)
    const [availableSuites, setAvailableSuites] = useState<string[]>([])
    const [availableMetrics, setAvailableMetrics] = useState<string[]>([])
    const [selectedSuite, setSelectedSuite] = useState<string>('')

    // Close map popup when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (mapPopup?.visible) {
                setMapPopup(null)
            }
        }

        document.addEventListener('click', handleClickOutside)
        return () => {
            document.removeEventListener('click', handleClickOutside)
        }
    }, [mapPopup])

    // Mock training data for demonstration - in real implementation, this would come from the API
    const mockTrainingData = {
        epochs: Array.from({ length: 5000 }, (_, i) => i + 1),
        rewards: Array.from({ length: 5000 }, (_, i) => {
            if (i < 1000) return 1 + (i / 1000) * 6
            if (i < 1500) return 7 - ((i - 1000) / 500) * 3
            return 2 + Math.sin(i / 100) * 2 + Math.random() * 0.5
        })
    }

    useEffect(() => {
        const loadPolicyData = async () => {
            if (!policyName) return

            try {
                setLoading(true)
                const [policyData, suites] = await Promise.all([
                    repo.getPolicyIds([policyName]).then(ids => {
                        if (!ids[policyName]) {
                            throw new Error(`Policy ${policyName} not found`)
                        }
                        return { id: ids[policyName], name: policyName }
                    }),
                    repo.getSuites()
                ])
                setPolicy(policyData)
                setAvailableSuites(suites)

                if (suites.length > 0) {
                    setSelectedSuite(suites[0])
                }
            } catch (err: any) {
                setError(`Failed to load policy: ${err.message}`)
            } finally {
                setLoading(false)
            }
        }

        loadPolicyData()
    }, [policyName, repo])

    // Load metrics when suite changes
    useEffect(() => {
        const loadMetrics = async () => {
            if (!selectedSuite) return

            try {
                const metrics = await repo.getMetrics(selectedSuite)
                setAvailableMetrics(metrics)

                // Set initial selected metrics if not already set
                if (metrics.length > 0) {
                    if (!selectedTab) setSelectedTab(metrics[0])
                    if (!selectedEvalTab) setSelectedEvalTab(metrics[0])
                }
            } catch (err: any) {
                console.error('Failed to load metrics:', err)
            }
        }

        loadMetrics()
    }, [selectedSuite, repo, selectedTab, selectedEvalTab])

    // Load heatmap data when suite or metric changes
    useEffect(() => {
        const loadHeatmapData = async () => {
            if (!policyName || !selectedSuite || !selectedTab) return

            try {
                const heatmapData = await repo.getEpochHeatmapData(selectedTab, selectedSuite)

                // Filter to only show the current policy
                if (heatmapData && Object.keys(heatmapData.cells).length > 0) {
                    // Try to find the exact policy name first, otherwise use the first available policy
                    const availablePolicies = Object.keys(heatmapData.cells)
                    const targetPolicy = availablePolicies.includes(policyName) ? policyName : availablePolicies[0]

                    const filteredHeatmapData = {
                        ...heatmapData,
                        cells: { [targetPolicy]: heatmapData.cells[targetPolicy] },
                        policyAverageScores: { [targetPolicy]: heatmapData.policyAverageScores[targetPolicy] || 0 }
                    }
                    setSampleRunsHeatmap(filteredHeatmapData)
                } else {
                    // If no data available, create empty heatmap
                    setSampleRunsHeatmap({
                        evalNames: heatmapData?.evalNames || [],
                        epochs: heatmapData?.epochs || [],
                        cells: {},
                        policyAverageScores: {},
                        evalAverageScores: heatmapData?.evalAverageScores || {},
                        evalMaxScores: heatmapData?.evalMaxScores || {}
                    })
                }
            } catch (err: any) {
                console.error('Failed to load sample runs heatmap:', err)
            }
        }

        loadHeatmapData()
    }, [policyName, selectedSuite, selectedTab, repo])

    // Load evaluation heatmap data
    useEffect(() => {
        const loadEvaluationHeatmap = async () => {
            if (!policyName || !selectedSuite || !selectedEvalTab) return

            try {
                const heatmapData = await repo.getHeatmapData(selectedEvalTab, selectedSuite)

                // Filter to only show the current policy
                if (heatmapData && policyName in heatmapData.cells) {
                    const filteredHeatmapData = {
                        ...heatmapData,
                        cells: { [policyName]: heatmapData.cells[policyName] },
                        policyAverageScores: { [policyName]: heatmapData.policyAverageScores[policyName] || 0 }
                    }
                    setEvaluationHeatmap(filteredHeatmapData)
                } else {
                    // If policy not found in heatmap data, create empty heatmap
                    setEvaluationHeatmap({
                        evalNames: heatmapData?.evalNames || [],
                        cells: {},
                        policyAverageScores: {},
                        evalAverageScores: heatmapData?.evalAverageScores || {},
                        evalMaxScores: heatmapData?.evalMaxScores || {}
                    })
                }
            } catch (err: any) {
                console.error('Failed to load evaluation heatmap:', err)
            }
        }

        loadEvaluationHeatmap()
    }, [policyName, selectedSuite, selectedEvalTab, repo])

    const getStatusClass = (status: string) => {
        switch (status.toLowerCase()) {
            case 'running':
                return 'running'
            case 'completed':
            case 'finished':
                return 'completed'
            case 'failed':
            case 'error':
                return 'failed'
            default:
                return 'running'
        }
    }

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString()
    }

    const getColorForValue = (value: number, maxValue: number = 100) => {
        const normalized = value / maxValue
        if (normalized < 0.3) return '#ff4444' // Red
        if (normalized < 0.6) return '#ffaa00' // Orange
        return '#44ff44' // Green
    }

    const getBluePurpleColor = (value: number, maxValue: number = 100) => {
        const normalized = value / maxValue
        const hue = 240 + normalized * 60 // Blue to purple
        const saturation = 70 + normalized * 30
        const lightness = 50 - normalized * 20
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`
    }

    const renderTrainingRewardChart = () => (
        <div className="training-reward-chart">
            <div className="metrics-section-title">Policy Performance</div>
            <Plot
                data={[
                    {
                        x: mockTrainingData.epochs,
                        y: mockTrainingData.rewards,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#007bff', width: 2 },
                        name: 'Policy Performance'
                    }
                ]}
                layout={{
                    height: 300,
                    margin: { t: 30, b: 50, l: 60, r: 30 },
                    xaxis: {
                        title: { text: 'epoch' },
                        range: [0, 5000],
                        tickmode: 'array',
                        tickvals: [1000, 2000, 3000, 4000, 5000],
                        ticktext: ['1k', '2k', '3k', '4k', '5k']
                    },
                    yaxis: {
                        title: { text: 'Performance' },
                        range: [1, 7]
                    },
                    showlegend: false,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
            />
        </div>
    )

    const renderSampleRunsGrid = () => (
        <div className="sample-runs-section">
            <div className="metrics-section-title">Sample Runs Used for Training</div>
            <div className="tabs-container">
                <div className="tabs">
                    <div style={{ overflowX: 'auto', whiteSpace: 'nowrap' }}>
                        {availableMetrics.map((metric) => (
                            <button
                                key={metric}
                                className={`tab ${selectedTab === metric ? 'active' : ''}`}
                                onClick={() => setSelectedTab(metric)}
                                style={{ display: 'inline-block', minWidth: 80 }}
                            >
                                {metric}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
            <div className="table-container">
                {sampleRunsHeatmap ? (
                    <div style={{ overflowX: 'auto' }}>
                        <table className="policy-evaluations-table" style={{
                            borderCollapse: 'collapse',
                            width: '100%',
                            fontSize: '12px'
                        }}>
                            <thead>
                                <tr>
                                    <th style={{
                                        padding: '8px',
                                        textAlign: 'left',
                                        borderBottom: '1px solid #ddd',
                                        fontWeight: 'bold',
                                        minWidth: '120px'
                                    }}>
                                        Tasks
                                    </th>
                                    {sampleRunsHeatmap.epochs.map((epoch) => (
                                        <th key={epoch} style={{
                                            padding: '8px',
                                            textAlign: 'center',
                                            borderBottom: '1px solid #ddd',
                                            fontWeight: 'bold',
                                            minWidth: '60px',
                                            fontSize: '11px'
                                        }}>
                                            {epoch >= 1000 ? `${epoch / 1000}k` : epoch}
                                        </th>
                                    ))}
                                    <th style={{
                                        padding: '8px',
                                        textAlign: 'center',
                                        borderBottom: '1px solid #ddd',
                                        minWidth: '30px'
                                    }}>
                                        C
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {sampleRunsHeatmap.evalNames.map((evalName, rowIndex) => (
                                    <tr key={evalName}>
                                        <td style={{
                                            padding: '8px',
                                            textAlign: 'left',
                                            borderBottom: '1px solid #eee',
                                            fontWeight: '500',
                                            fontSize: '11px'
                                        }}>
                                            {evalName}
                                        </td>
                                        {sampleRunsHeatmap.epochs.map((epoch) => {
                                            // Get the cell for the current policy, eval, and epoch
                                            const policyUri = Object.keys(sampleRunsHeatmap.cells)[0] // Should be the current policy
                                            const cell = sampleRunsHeatmap.cells[policyUri]?.[evalName]?.[epoch.toString()]
                                            if (!cell) {
                                                return <td key={epoch} style={{
                                                    padding: '4px',
                                                    textAlign: 'center',
                                                    backgroundColor: '#f8f9fa',
                                                    borderBottom: '1px solid #eee',
                                                    minWidth: '60px',
                                                    height: '30px'
                                                }}></td>
                                            }
                                            const value = cell?.value || 0

                                            // Color coding based on performance
                                            let backgroundColor = '#f8f9fa' // Default grey
                                            if (value > 0) {
                                                if (value >= 100) {
                                                    backgroundColor = '#4caf50' // Green for excellent performance
                                                } else if (value >= 50) {
                                                    backgroundColor = '#ff9800' // Orange for moderate performance
                                                } else {
                                                    backgroundColor = '#f44336' // Red for poor performance
                                                }
                                            }

                                            return (
                                                <td
                                                    key={epoch}
                                                    style={{
                                                        backgroundColor,
                                                        color: value > 0 ? 'white' : '#666',
                                                        cursor: 'pointer',
                                                        textAlign: 'center',
                                                        padding: '4px',
                                                        fontSize: '10px',
                                                        fontWeight: 'bold',
                                                        borderBottom: '1px solid #eee',
                                                        minWidth: '60px',
                                                        height: '30px',
                                                        transition: 'all 0.2s ease'
                                                    }}
                                                    onMouseEnter={() => setHoveredCell({ x: epoch, y: rowIndex })}
                                                    onMouseLeave={() => setHoveredCell(null)}
                                                    onClick={(e) => {
                                                        e.stopPropagation()
                                                        setMapPopup({ x: epoch, y: rowIndex, visible: true })
                                                    }}
                                                    title={`${evalName} - Epoch ${epoch}: ${value.toFixed(1)}`}
                                                >
                                                    {value > 0 ? value.toFixed(0) : ''}
                                                </td>
                                            )
                                        })}
                                        <td style={{
                                            padding: '8px',
                                            textAlign: 'center',
                                            borderBottom: '1px solid #eee'
                                        }}>
                                            <div style={{
                                                width: '16px',
                                                height: '16px',
                                                borderRadius: '50%',
                                                backgroundColor: '#9e9e9e',
                                                color: 'white',
                                                fontSize: '10px',
                                                fontWeight: 'bold',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                margin: '0 auto'
                                            }}>
                                                C
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="loading-container">
                        <div>Loading policy evaluations...</div>
                    </div>
                )}
                {mapPopup?.visible && (
                    <div
                        className="map-popup"
                        style={{
                            left: mapPopup.x * 32 + 50,
                            top: mapPopup.y * 32 + 50,
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="map-popup-header">
                            <div className="map-popup-title">Map: Simple</div>
                            <div className="map-popup-upload">📤</div>
                        </div>
                        <div className="map-preview">
                            <div>Simple maze environment with red objects</div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )

    const renderEvaluationScoresGrid = () => (
        <div className="evaluation-scores-section">
            <div className="metrics-section-title">Evaluation Scores During Training</div>
            <div className="tabs-container">
                <div className="tabs">
                    <div style={{ overflowX: 'auto', whiteSpace: 'nowrap' }}>
                        {availableMetrics.map((metric) => (
                            <button
                                key={metric}
                                className={`tab ${selectedEvalTab === metric ? 'active' : ''}`}
                                onClick={() => setSelectedEvalTab(metric)}
                                style={{ display: 'inline-block', whiteSpace: 'nowrap' }}
                            >
                                {metric}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
            <div className="grid-container">
                {evaluationHeatmap ? (
                    <div style={{ display: 'flex', alignItems: 'flex-start', overflowX: 'scroll' }}>
                        <table className="evaluation-table">
                            <thead>
                                <tr>
                                    <th className="evaluation-header" style={{
                                        fontWeight: 'bold',
                                        background: '#e3f2fd',
                                        minWidth: '100px'
                                    }}>
                                        Tasks
                                    </th>
                                    <th
                                        className="evaluation-header"
                                        style={{
                                            fontWeight: 'bold',
                                            background: '#e3f2fd',
                                            fontSize: '10px',
                                            textAlign: 'center',
                                            padding: '4px 2px',
                                            borderBottom: '2px solid #007bff',
                                            minWidth: '100px'
                                        }}
                                    >
                                        {selectedEvalTab}
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {evaluationHeatmap.evalNames.map((evalName, taskIndex) => (
                                    <tr key={evalName}>
                                        <td className="evaluation-row-header" style={{
                                            fontSize: '11px',
                                            minWidth: '100px'
                                        }}>
                                            {evalName}
                                        </td>
                                        {(() => {
                                            // Show the value for the current policy and evaluation task
                                            const policyUri = Object.keys(evaluationHeatmap.cells)[0] // Should be the current policy
                                            const cell = evaluationHeatmap.cells[policyUri]?.[evalName]
                                            const value = cell?.value || 0

                                            return (
                                                <td
                                                    className="evaluation-cell"
                                                    style={{
                                                        flexDirection: 'column',
                                                        justifyContent: 'center',
                                                        alignItems: 'center',
                                                        gap: '1px',
                                                        padding: '2px',
                                                        backgroundColor: '#f0f8ff',
                                                        minWidth: '80px',
                                                        minHeight: '30px'
                                                    }}
                                                >
                                                    <div
                                                        style={{
                                                            width: '8px',
                                                            height: '8px',
                                                            backgroundColor: getColorForValue(value),
                                                            borderRadius: '1px'
                                                        }}
                                                    />
                                                    <div style={{ fontSize: '8px', color: '#666' }}>
                                                        {value.toFixed(1)}
                                                    </div>
                                                </td>
                                            )
                                        })()}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <div className="completion-indicators">
                            {evaluationHeatmap.evalNames.map((_, i) => (
                                <div key={i} className="completion-indicator">
                                    <div style={{
                                        width: '16px',
                                        height: '16px',
                                        borderRadius: '50%',
                                        backgroundColor: '#e0e0e0',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        fontSize: '10px',
                                        fontWeight: 'bold',
                                        color: '#666'
                                    }}>
                                        C
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="loading-container">
                        <div>Loading evaluation scores data...</div>
                    </div>
                )}
            </div>
        </div>
    )

    const renderAdditionalStats = () => (
        <div className="additional-stats-section">
            <div className="stat-chart">
                <div className="metrics-section-title">Policy Metrics</div>
                <Plot
                    data={[
                        {
                            x: mockTrainingData.epochs,
                            y: mockTrainingData.rewards,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#007bff', width: 2 },
                            name: 'Policy Performance'
                        }
                    ]}
                    layout={{
                        height: 250,
                        margin: { t: 30, b: 50, l: 60, r: 30 },
                        xaxis: {
                            title: { text: 'epoch' },
                            range: [0, 5000],
                            tickmode: 'array',
                            tickvals: [1000, 2000, 3000, 4000, 5000],
                            ticktext: ['1k', '2k', '3k', '4k', '5k']
                        },
                        yaxis: {
                            title: { text: 'Performance' },
                            range: [1, 7]
                        },
                        showlegend: false,
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)'
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: '100%' }}
                />
            </div>
            <div className="stat-chart">
                <div className="metrics-section-title">Additional Metrics</div>
                <Plot
                    data={[
                        {
                            x: mockTrainingData.epochs,
                            y: mockTrainingData.rewards,
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#007bff', width: 2 },
                            name: 'Policy Performance'
                        }
                    ]}
                    layout={{
                        height: 250,
                        margin: { t: 30, b: 50, l: 60, r: 30 },
                        xaxis: {
                            title: { text: 'epoch' },
                            range: [0, 5000],
                            tickmode: 'array',
                            tickvals: [1000, 2000, 3000, 4000, 5000],
                            ticktext: ['1k', '2k', '3k', '4k', '5k']
                        },
                        yaxis: {
                            title: { text: 'Performance' },
                            range: [1, 7]
                        },
                        showlegend: false,
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)'
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: '100%' }}
                />
            </div>
        </div>
    )

    if (loading) {
        return (
            <div className="policy-view-container">
                <style>{POLICY_VIEW_CSS}</style>
                <div className="policy-view-content">
                    <div className="loading-container">
                        <div>Loading policy data...</div>
                    </div>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="policy-view-container">
                <style>{POLICY_VIEW_CSS}</style>
                <div className="policy-view-content">
                    <div className="error-container">
                        <div>{error}</div>
                    </div>
                </div>
            </div>
        )
    }

    if (!policy) {
        return (
            <div className="policy-view-container">
                <style>{POLICY_VIEW_CSS}</style>
                <div className="policy-view-content">
                    <div className="error-container">
                        <div>Policy not found</div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="policy-view-container">
            <style>{POLICY_VIEW_CSS}</style>
            <div className="policy-view-content">
                <div className="policy-header">
                    <div className="breadcrumb">
                        <Link to="/policies">Policies</Link>
                        <span className="breadcrumb-separator">›</span>
                        <span>{policy.name}</span>
                    </div>
                    <h1 className="policy-title">{policy.name}</h1>
                    <div className="policy-meta">
                        <div className="policy-meta-item">
                            <span>Policy ID:</span>
                            <span>{policy.id}</span>
                        </div>
                        <div className="policy-meta-item">
                            <span>Status:</span>
                            <span className="policy-status active">Active</span>
                        </div>
                    </div>
                </div>

                <div className="metrics-section">
                    {availableSuites.length > 0 && (
                        <div style={{ marginBottom: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
                            <label style={{ marginRight: '10px', fontWeight: '500' }}>Suite:</label>
                            <select
                                value={selectedSuite}
                                onChange={(e) => setSelectedSuite(e.target.value)}
                                style={{ padding: '5px 10px', borderRadius: '4px', border: '1px solid #ddd' }}
                            >
                                {availableSuites.map((suite) => (
                                    <option key={suite} value={suite}>
                                        {suite}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                    {renderTrainingRewardChart()}
                    {renderSampleRunsGrid()}
                    {renderEvaluationScoresGrid()}
                    {renderAdditionalStats()}
                </div>
            </div>
        </div>
    )
}
