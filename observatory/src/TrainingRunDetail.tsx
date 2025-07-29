import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { TrainingRun, PolicyHeatmapData, TrainingRunPolicy, Repo } from './repo'
import { MapViewer } from './MapViewer'
import { EvalSelector } from './components/EvalSelector'
import { MetricSelector } from './components/MetricSelector'
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
  const [heatmapData, setHeatmapData] = useState<PolicyHeatmapData | null>(null)
  const [evalNames, setEvalNames] = useState<Set<string>>(new Set())
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([])
  const [trainingRunPolicies, setTrainingRunPolicies] = useState<TrainingRunPolicy[]>([])

  // Selection state
  const [selectedEvalNames, setSelectedEvalNames] = useState<Set<string>>(new Set())
  const [selectedMetric, setSelectedMetric] = useState<string>('')

  // UI state
  const [loading, setLoading] = useState({
    initial: true,
    policies: false,
    evalNames: false,
    metrics: false,
    heatmap: false
  })
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
        setLoading(prev => ({ ...prev, initial: true }))
        const [runData, userResponse] = await Promise.all([
          repo.getTrainingRun(runId),
          repo.whoami().catch(() => ({ user_email: '' })),
        ])

        setTrainingRun(runData)
        setCurrentUser(userResponse.user_email)
        setError(null)
      } catch (err: any) {
        setError(`Failed to load training run: ${err.message}`)
      } finally {
        setLoading(prev => ({ ...prev, initial: false }))
      }
    }

    initializeData()
  }, [runId, repo])

  // Load training run policies with epoch information
  useEffect(() => {
    const loadPolicies = async () => {
      if (!runId) return

      try {
        setLoading(prev => ({ ...prev, policies: true }))
        setError(null)
        const policiesData = await repo.getTrainingRunPolicies(runId)
        setTrainingRunPolicies(policiesData)
      } catch (err) {
        setError(`Failed to load policies: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setTrainingRunPolicies([])
      } finally {
        setLoading(prev => ({ ...prev, policies: false }))
      }
    }

    loadPolicies()
  }, [runId, repo])

  // Load eval names when training run is loaded
  useEffect(() => {
    const loadEvalNames = async () => {
      if (!runId) return

      try {
        setLoading(prev => ({ ...prev, evalNames: true }))
        setError(null)
        const evalNamesData = await repo.getEvalNames({
          training_run_ids: [runId],
          run_free_policy_ids: []
        })
        setEvalNames(evalNamesData)

        // Clear eval selections that are no longer valid
        setSelectedEvalNames(prev => new Set([...prev].filter(evalName => evalNamesData.has(evalName))))
      } catch (err) {
        setError(`Failed to load eval names: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setEvalNames(new Set())
        setSelectedEvalNames(new Set())
      } finally {
        setLoading(prev => ({ ...prev, evalNames: false }))
      }
    }

    loadEvalNames()
  }, [runId, repo])

  // Load available metrics when evaluations are selected
  useEffect(() => {
    const loadMetrics = async () => {
      if (!runId || selectedEvalNames.size === 0) {
        setAvailableMetrics([])
        setSelectedMetric('')
        return
      }

      try {
        setLoading(prev => ({ ...prev, metrics: true }))
        setError(null)
        const metricsData = await repo.getAvailableMetrics({
          training_run_ids: [runId],
          run_free_policy_ids: [],
          eval_names: Array.from(selectedEvalNames)
        })
        setAvailableMetrics(metricsData)

        // Clear metric selection if it's no longer available
        if (selectedMetric && !metricsData.includes(selectedMetric)) {
          setSelectedMetric('')
        }
      } catch (err) {
        setError(`Failed to load metrics: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setAvailableMetrics([])
        setSelectedMetric('')
      } finally {
        setLoading(prev => ({ ...prev, metrics: false }))
      }
    }

    loadMetrics()
  }, [runId, selectedEvalNames, selectedMetric, repo])

  // Load heatmap data when parameters change
  useEffect(() => {
    const loadHeatmapData = async () => {
      if (!runId || selectedEvalNames.size === 0 || !selectedMetric) return

      try {
        setLoading(prev => ({ ...prev, heatmap: true }))
        setError(null)
        const heatmapData = await repo.generateTrainingRunHeatmap(runId, {
          eval_names: Array.from(selectedEvalNames),
          metric: selectedMetric
        })
        setHeatmapData(heatmapData)
      } catch (err: any) {
        setError(`Failed to load heatmap data: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setHeatmapData(null)
      } finally {
        setLoading(prev => ({ ...prev, heatmap: false }))
      }
    }

    loadHeatmapData()
  }, [runId, selectedEvalNames, selectedMetric, repo])

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

    const policies = heatmapData.policyNames
    const evalNames = heatmapData.evalNames

    if (policies.length === 0) {
      return <div style={{ textAlign: 'center', padding: '20px' }}>No policies found for this training run.</div>
    }

    if (evalNames.length === 0) {
      return <div style={{ textAlign: 'center', padding: '20px' }}>No evaluations found for selected criteria.</div>
    }

    if (trainingRunPolicies.length === 0) {
      return <div style={{ textAlign: 'center', padding: '20px' }}>No policy epoch data available. Loading...</div>
    }

    // Sort policies by epoch information instead of parsing names
    const policyNameToEpoch = new Map<string, number>()
    trainingRunPolicies.forEach((policy) => {
      // Use epoch_start for sorting, fallback to 0 if null
      policyNameToEpoch.set(policy.policy_name, policy.epoch_end ?? 0)
    })

    // Sort policies by epoch_start, then by name for consistent ordering
    const sortedPolicies = policies
      .filter(policy => policyNameToEpoch.has(policy)) // Only include policies we have epoch data for
      .sort((a, b) => {
        const epochA = policyNameToEpoch.get(a) ?? 0
        const epochB = policyNameToEpoch.get(b) ?? 0
        if (epochA !== epochB) {
          return epochA - epochB
        }
        return a.localeCompare(b) // Secondary sort by name
      })

    const shortNameToEvalName = new Map<string, string>()
    evalNames.forEach((evalName) => {
      shortNameToEvalName.set(getShortName(evalName), evalName)
    })
    const sortedShortNames = [...shortNameToEvalName.keys()].sort((a, b) => b.localeCompare(a))

    const xLabels = sortedPolicies.map(policy => {
      const epoch = policyNameToEpoch.get(policy) ?? 0
      return `Step ${epoch}` // Use epoch as the label
    })
    const yLabels = sortedShortNames

    const z = yLabels.map((shortName) =>
      sortedPolicies.map((policy) => {
        const evalName = shortNameToEvalName.get(shortName)!
        const cell = heatmapData.cells[policy]?.[evalName]
        return cell ? cell.value : 0
      })
    )

    const onHover = (event: any) => {
      if (!event.points?.[0]) return

      const epochLabel = event.points[0].x
      const shortName = event.points[0].y
      const evalName = shortNameToEvalName.get(shortName)!

      // Find the policy by epoch label index
      const policyIndex = xLabels.indexOf(epochLabel)
      if (policyIndex >= 0 && policyIndex < sortedPolicies.length) {
        const policyUri = sortedPolicies[policyIndex]
        setLastHoveredCell({ policyUri, evalName })
        setSelectedCellIfNotLocked({ policyUri, evalName })
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

  if (loading.initial) {
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

        <div className="heatmap-controls">
          <div className="control-section">
            <h3>Evaluation Selection</h3>
            <EvalSelector
              evalNames={evalNames}
              selectedEvalNames={selectedEvalNames}
              onSelectionChange={setSelectedEvalNames}
              loading={loading.evalNames}
            />
          </div>

          <div className="control-section">
            <h3>Metric Selection</h3>
            <MetricSelector
              metrics={availableMetrics}
              selectedMetric={selectedMetric}
              onSelectionChange={setSelectedMetric}
              loading={loading.metrics}
              disabled={selectedEvalNames.size === 0}
            />
          </div>
        </div>

        {loading.heatmap && (
          <div className="loading-container">
            <div>Loading heatmap...</div>
          </div>
        )}

        {!loading.heatmap && heatmapData && renderHeatmap()}

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
