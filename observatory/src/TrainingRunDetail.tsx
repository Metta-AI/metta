import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { TrainingRun, HeatmapData, Repo } from './repo'
import { MapViewer } from './MapViewer'
import { SuiteTabs } from './SuiteTabs'
import { TagEditor } from './TagEditor'
import { DescriptionEditor } from './DescriptionEditor'

const TRAINING_RUN_DETAIL_CSS = `
.training-run-detail-container {
  padding: 20px;
  background: #f8f9fa;
  min-height: calc(100vh - 60px);
}

.training-run-detail-content {
  max-width: 1200px;
  margin: 0 auto;
  background: #fff;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.training-run-header {
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

.training-run-title {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 24px;
  font-weight: 600;
}

.training-run-meta {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #666;
}

.training-run-meta-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.training-run-status {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
}

.training-run-status.running {
  background: #e3f2fd;
  color: #1976d2;
}

.training-run-status.completed {
  background: #e8f5e8;
  color: #2e7d32;
}

.training-run-status.failed {
  background: #ffebee;
  color: #c62828;
}


.heatmap-controls {
  display: grid;
  gridTemplateColumns: 1fr 1fr;
  gap: 20px;
  marginTop: 30px;
  marginBottom: 30px;
}

.control-section {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.control-section h3 {
  margin: 0 0 15px 0;
  fontSize: 16px;
  fontWeight: 600;
  color: #333;
}

.control-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 15px;
}

.control-label {
  color: #666;
  fontSize: 14px;
  minWidth: 120px;
}

.control-select {
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #ddd;
  fontSize: 14px;
  flex: 1;
  backgroundColor: #fff;
  cursor: pointer;
}

.loading-container,
.error-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
  color: #666;
  text-align: center;
}

.error-container {
  color: #c62828;
}

.training-run-description-section {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.training-run-description-section-header {
  margin-bottom: 10px;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.training-run-tags-section {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.training-run-tags-section-header {
  margin-bottom: 10px;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}
`

interface TrainingRunDetailProps {
  repo: Repo
}

const getShortName = (evalName: string) => {
  if (evalName === 'Overall') return evalName
  return evalName.split('/').pop() || evalName
}

export function TrainingRunDetail({ repo }: TrainingRunDetailProps) {
  const { runId } = useParams<{ runId: string }>()

  // Data state
  const [trainingRun, setTrainingRun] = useState<TrainingRun | null>(null)
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null)
  const [metrics, setMetrics] = useState<string[]>([])
  const [suites, setSuites] = useState<string[]>([])

  // UI state
  const [selectedMetric, setSelectedMetric] = useState<string>('reward')
  const [selectedSuite, setSelectedSuite] = useState<string>('navigation')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [selectedCell, setSelectedCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)
  const [lastHoveredCell, setLastHoveredCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)
  const [saving, setSaving] = useState(false)
  const [currentUser, setCurrentUser] = useState<string | null>(null)

  // Load training run and initial data
  useEffect(() => {
    const initializeData = async () => {
      if (!runId) return

      try {
        setLoading(true)
        const [runData, suitesData, userResponse] = await Promise.all([
          repo.getTrainingRun(runId),
          repo.getSuites(),
          repo.whoami().catch(() => ({ user_email: '' })),
        ])

        setTrainingRun(runData)
        setSuites(suitesData)
        setSelectedSuite(suitesData[0])
        setCurrentUser(userResponse.user_email)
        setError(null)
      } catch (err: any) {
        setError(`Failed to load training run: ${err.message}`)
      } finally {
        setLoading(false)
      }
    }

    initializeData()
  }, [runId, repo])

  // Load metrics when suite changes
  useEffect(() => {
    const loadSuiteData = async () => {
      if (!selectedSuite) return

      try {
        const metricsData = await repo.getAllMetrics()
        setMetrics(metricsData)
      } catch (err: any) {
        setError(`Failed to load suite data: ${err.message}`)
      }
    }

    loadSuiteData()
  }, [selectedSuite, repo])

  // Load heatmap data when parameters change
  useEffect(() => {
    const loadHeatmapData = async () => {
      if (!runId || !selectedSuite || !selectedMetric) return

      try {
        const heatmapData = await repo.getTrainingRunHeatmapData(
          runId,
          selectedMetric,
          selectedSuite,
        )
        setHeatmapData(heatmapData)
      } catch (err: any) {
        setError(`Failed to load heatmap data: ${err.message}`)
      }
    }

    loadHeatmapData()
  }, [runId, selectedSuite, selectedMetric, repo])

  const setSelectedCellIfNotLocked = (cell: {
    policyUri: string
    evalName: string
  }) => {
    if (!isViewLocked) {
      setSelectedCell(cell)
    }
  }

  const openReplayUrl = (policyUri: string, evalName: string) => {
    const evalData = heatmapData?.cells[policyUri]?.[evalName]
    if (!evalData?.replayUrl) return

    const replay_url_prefix = 'https://metta-ai.github.io/metta/?replayUrl='
    window.open(replay_url_prefix + evalData.replayUrl, '_blank')
  }

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked)
  }

  const handleReplayClick = () => {
    if (selectedCell) {
      openReplayUrl(selectedCell.policyUri, selectedCell.evalName)
    }
  }

  const getStatusClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'running'
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      default:
        return ''
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const canEditRun = (run: TrainingRun) => {
    return Boolean(currentUser && run.user_id === currentUser)
  }

  const handleDescriptionChange = async (newDescription: string) => {
    if (!runId) return

    setSaving(true)
    try {
      const updatedRun = await repo.updateTrainingRunDescription(runId, newDescription)
      setTrainingRun(updatedRun)
    } finally {
      setSaving(false)
    }
  }

  const handleTagsChange = async (newTags: string[]) => {
    if (!runId || !trainingRun) return

    setSaving(true)
    try {
      const updatedRun = await repo.updateTrainingRunTags(runId, newTags)
      setTrainingRun(updatedRun)
    } finally {
      setSaving(false)
    }
  }

  // Create the modified heatmap with policies on X-axis and evals on Y-axis
  const renderHeatmap = () => {
    if (!heatmapData) return null

    const policies = Object.keys(heatmapData.cells)
    const evalNames = heatmapData.evalNames

    // Sort policies by version number
    const policyVersionToPolicy = new Map<string, string>()
    policies.forEach((policy) => {
      policyVersionToPolicy.set(policy.split(':v')[1], policy)
    })
    const sortedPolicyVersions = [...policyVersionToPolicy.keys()].sort((a, b) => parseInt(a) - parseInt(b))

    const shortNameToEvalName = new Map<string, string>()
    evalNames.forEach((evalName) => {
      shortNameToEvalName.set(getShortName(evalName), evalName)
    })
    const sortedShortNames = [...shortNameToEvalName.keys()].sort((a, b) => b.localeCompare(a))

    const xLabels = sortedPolicyVersions
    const yLabels = sortedShortNames

    const z = yLabels.map((shortName) =>
      xLabels.map((policyVersion) => {
        const evalName = shortNameToEvalName.get(shortName)!
        const policy = policyVersionToPolicy.get(policyVersion)!
        const cell = heatmapData.cells[policy]?.[evalName]
        return cell ? cell.value : 0
      })
    )

    const onHover = (event: any) => {
      if (!event.points?.[0]) return

      const policyUri = event.points[0].x
      const shortName = event.points[0].y
      const evalName = shortNameToEvalName.get(shortName)!

      setLastHoveredCell({ policyUri, evalName })
      setSelectedCellIfNotLocked({ policyUri, evalName })
    }

    const onDoubleClick = () => {
      if (lastHoveredCell) {
        openReplayUrl(lastHoveredCell.policyUri, lastHoveredCell.evalName)
      }
    }

    const plotData: Plotly.Data = {
      z,
      x: xLabels,
      y: yLabels,
      type: 'heatmap',
      colorscale: 'Viridis',
      colorbar: {
        title: {
          text: selectedMetric,
        },
      },
    }

    return (
      <Plot
        data={[plotData]}
        layout={{
          title: {
            text: `Training Run: ${trainingRun?.name} - ${selectedMetric}`,
            font: { size: 20 },
          },
          height: 600,
          width: 1000,
          margin: { t: 50, b: 150, l: 150, r: 50 },
          xaxis: {
            tickangle: -45,
            title: { text: 'Policies (ordered by epoch)' },
          },
          yaxis: {
            tickangle: 0,
            title: { text: 'Evaluations' },
          },
        }}
        style={{
          margin: '0 auto',
          display: 'block',
        }}
        onHover={onHover}
        onDoubleClick={onDoubleClick}
      />
    )
  }

  const selectedCellData = selectedCell ? heatmapData?.cells[selectedCell.policyUri]?.[selectedCell.evalName] : null
  const selectedEval = selectedCellData?.evalName ?? null
  const selectedReplayUrl = selectedCellData?.replayUrl ?? null

  if (loading) {
    return (
      <div className="training-run-detail-container">
        <style>{TRAINING_RUN_DETAIL_CSS}</style>
        <div className="training-run-detail-content">
          <div className="loading-container">
            <div>Loading training run...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="training-run-detail-container">
        <style>{TRAINING_RUN_DETAIL_CSS}</style>
        <div className="training-run-detail-content">
          <div className="error-container">
            <div>{error}</div>
          </div>
        </div>
      </div>
    )
  }

  if (!trainingRun) {
    return (
      <div className="training-run-detail-container">
        <style>{TRAINING_RUN_DETAIL_CSS}</style>
        <div className="training-run-detail-content">
          <div className="error-container">
            <div>Training run not found</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-run-detail-container">
      <style>{TRAINING_RUN_DETAIL_CSS}</style>
      <div className="training-run-detail-content">
        <div className="training-run-header">
          <div className="breadcrumb">
            <Link to="/training-runs">Training Runs</Link>
            <span className="breadcrumb-separator">›</span>
            <span>{trainingRun.name}</span>
          </div>
          <h1 className="training-run-title">{trainingRun.name}</h1>
          <div className="training-run-meta">
            <div className="training-run-meta-item">
              <span>Status:</span>
              <span className={`training-run-status ${getStatusClass(trainingRun.status)}`}>{trainingRun.status}</span>
            </div>
            <div className="training-run-meta-item">
              <span>Created:</span>
              <span>{formatDate(trainingRun.created_at)}</span>
            </div>
            {trainingRun.finished_at && (
              <div className="training-run-meta-item">
                <span>Finished:</span>
                <span>{formatDate(trainingRun.finished_at)}</span>
              </div>
            )}
            <div className="training-run-meta-item">
              <span>User:</span>
              <span>{trainingRun.user_id}</span>
            </div>
          </div>

          <div className="training-run-description-section">
            <div className="training-run-description-section-header">
              <strong>Description:</strong>
            </div>
            <DescriptionEditor
              description={trainingRun.description}
              canEdit={canEditRun(trainingRun)}
              onDescriptionChange={handleDescriptionChange}
              onError={setError}
              disabled={saving}
              compact={false}
              placeholder="Enter a description for this training run..."
            />
          </div>

          <div className="training-run-tags-section">
            <div className="training-run-tags-section-header">
              <strong>Tags:</strong>
            </div>
            <TagEditor
              tags={trainingRun.tags}
              canEdit={canEditRun(trainingRun)}
              onTagsChange={handleTagsChange}
              onError={setError}
              disabled={saving}
              compact={false}
            />
          </div>
        </div>

        <SuiteTabs suites={suites} selectedSuite={selectedSuite} onSuiteChange={setSelectedSuite} />

        {heatmapData && renderHeatmap()}

        <div className="heatmap-controls">
          <div className="control-section">
            <h3>Heatmap Controls</h3>
            <div className="control-row">
              <div className="control-label">Heatmap Metric</div>
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="control-select"
              >
                {metrics.map((metric) => (
                  <option key={metric} value={metric}>
                    {metric}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <MapViewer
          selectedEval={selectedEval}
          isViewLocked={isViewLocked}
          selectedReplayUrl={selectedReplayUrl}
          onToggleLock={toggleLock}
          onReplayClick={handleReplayClick}
        />
      </div>
    </div>
  )
}
