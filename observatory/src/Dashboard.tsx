import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { DashboardState, PolicyScorecardData, Repo, SavedDashboard, SavedDashboardCreate, UnifiedPolicyInfo } from './repo'
import { PolicySelector } from './components/PolicySelector'
import { EvalSelector } from './components/EvalSelector'
import { TrainingRunPolicySelector } from './components/TrainingRunPolicySelector'
import { MetricSelector } from './components/MetricSelector'
import { Scorecard } from './Scorecard'
import styles from './Dashboard.module.css'
import { MapViewer } from './MapViewer'
import { SaveDashboardModal } from './SaveDashboardModal'
import { METTASCOPE_REPLAY_URL } from './constants'

interface DashboardProps {
  repo: Repo
}

export function Dashboard({ repo }: DashboardProps) {
  const [searchParams, setSearchParams] = useSearchParams()
  // Data state
  const [policies, setPolicies] = useState<UnifiedPolicyInfo[]>([])
  const [evalNames, setEvalNames] = useState<Set<string>>(new Set())
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([])
  const [scorecardData, setScorecardData] = useState<PolicyScorecardData | null>(null)

  // Selection state
  const [selectedTrainingRunIds, setSelectedTrainingRunIds] = useState<string[]>([])
  const [selectedRunFreePolicyIds, setSelectedRunFreePolicyIds] = useState<string[]>([])
  const [selectedEvalNames, setSelectedEvalNames] = useState<Set<string>>(new Set())
  const [trainingRunPolicySelector, setTrainingRunPolicySelector] = useState<'latest' | 'best'>('latest')
  const [selectedMetric, setSelectedMetric] = useState<string>('reward')

  // UI state
  const [loading, setLoading] = useState({
    policies: false,
    evalCategories: false,
    metrics: false,
    scorecard: false,
  })
  const [error, setError] = useState<string | null>(null)
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [selectedCell, setSelectedCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)
  const [controlsExpanded, setControlsExpanded] = useState<boolean>(true)

  // Save dashboard modal state
  const [showSaveModal, setShowSaveModal] = useState(false)

  // Dashboard metadata state
  const [savedDashboard, setSavedDashboard] = useState<SavedDashboard | null>(null)

  // Load policies on mount
  useEffect(() => {
    const loadPolicies = async () => {
      try {
        setLoading((prev) => ({ ...prev, policies: true }))
        setError(null)
        const response = await repo.getPolicies()
        setPolicies(response.policies)
      } catch (err) {
        setError(`Failed to load policies: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setPolicies([])
      } finally {
        setLoading((prev) => ({ ...prev, policies: false }))
      }
    }

    loadPolicies()
  }, [])

  // Load eval names when training runs or policies are selected
  useEffect(() => {
    const loadEvalNames = async () => {
      if (selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0) {
        setEvalNames(new Set())
        setSelectedEvalNames(new Set())
        return
      }

      try {
        setLoading((prev) => ({ ...prev, evalCategories: true }))
        setError(null)
        const evalNamesData = await repo.getEvalNames({
          training_run_ids: selectedTrainingRunIds,
          run_free_policy_ids: selectedRunFreePolicyIds,
        })
        setEvalNames(evalNamesData)

        // Clear eval selections that are no longer valid
        setSelectedEvalNames(new Set([...selectedEvalNames].filter((evalName) => evalNamesData.has(evalName))))
      } catch (err) {
        setError(`Failed to load eval names: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setEvalNames(new Set())
        setSelectedEvalNames(new Set())
      } finally {
        setLoading((prev) => ({ ...prev, evalCategories: false }))
      }
    }

    loadEvalNames()
  }, [selectedTrainingRunIds, selectedRunFreePolicyIds])

  // Load available metrics when training runs/policies and evaluations are selected
  useEffect(() => {
    const loadMetrics = async () => {
      if (
        (selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0) ||
        selectedEvalNames.size === 0
      ) {
        setAvailableMetrics([])
        setSelectedMetric('')
        return
      }

      try {
        setLoading((prev) => ({ ...prev, metrics: true }))
        setError(null)
        const metricsData = await repo.getAvailableMetrics({
          training_run_ids: selectedTrainingRunIds,
          run_free_policy_ids: selectedRunFreePolicyIds,
          eval_names: Array.from(selectedEvalNames),
        })
        setAvailableMetrics(metricsData)

        // Clear metric selection if it's no longer available
        if (selectedMetric && !metricsData.includes(selectedMetric)) {
          setSelectedMetric('')
        }
        if (metricsData.length > 0 && !selectedMetric) {
          if (metricsData.includes('reward')) {
            setSelectedMetric('reward')
          } else {
            setSelectedMetric(metricsData[0])
          }
        }
      } catch (err) {
        setError(`Failed to load metrics: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setAvailableMetrics([])
        setSelectedMetric('')
      } finally {
        setLoading((prev) => ({ ...prev, metrics: false }))
      }
    }

    loadMetrics()
  }, [selectedTrainingRunIds, selectedRunFreePolicyIds, selectedEvalNames])

  const generateScorecard = async (
    selectedTrainingRunIds: string[],
    selectedRunFreePolicyIds: string[],
    selectedEvalNames: Set<string>,
    selectedMetric: string
  ) => {
    if (
      (selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0) ||
      selectedEvalNames.size === 0 ||
      !selectedMetric
    ) {
      setError('Please select training runs/policies, evaluations, and a metric before generating the scorecard.')
      return
    }

    try {
      setLoading((prev) => ({ ...prev, scorecard: true }))
      setError(null)
      const scorecardResult = await repo.generatePolicyScorecard({
        training_run_ids: selectedTrainingRunIds,
        run_free_policy_ids: selectedRunFreePolicyIds,
        eval_names: Array.from(selectedEvalNames),
        training_run_policy_selector: trainingRunPolicySelector,
        metric: selectedMetric,
      })
      setScorecardData(scorecardResult)
      setControlsExpanded(false)
    } catch (err) {
      setError(`Failed to generate scorecard: ${err instanceof Error ? err.message : 'Unknown error'}`)
      setScorecardData(null)
    } finally {
      setLoading((prev) => ({ ...prev, scorecard: false }))
    }
  }

  const generateScorecardCallback = async () => {
    await generateScorecard(selectedTrainingRunIds, selectedRunFreePolicyIds, selectedEvalNames, selectedMetric)
  }

  const canGenerateScorecard =
    (selectedTrainingRunIds.length > 0 || selectedRunFreePolicyIds.length > 0) &&
    selectedEvalNames.size > 0 &&
    selectedMetric !== '' &&
    !Object.values(loading).some(Boolean)

  const openReplayUrl = (policyName: string, evalName: string) => {
    const cell = scorecardData?.cells[policyName]?.[evalName]
    if (!cell?.replayUrl) return

    const replay_url_prefix = `${METTASCOPE_REPLAY_URL}/?replayUrl=`
    window.open(replay_url_prefix + cell.replayUrl, '_blank')
  }

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked)
  }

  const handleReplayClick = () => {
    if (selectedCell) {
      openReplayUrl(selectedCell.policyUri, selectedCell.evalName)
    }
  }

  const selectedCellData = selectedCell ? scorecardData?.cells[selectedCell.policyUri]?.[selectedCell.evalName] : null
  const selectedEval = selectedCellData?.evalName ?? null
  const selectedReplayUrl = selectedCellData?.replayUrl ?? null

  // Dashboard state management functions
  const getDashboardState = () => {
    return {
      selectedTrainingRunIds,
      selectedRunFreePolicyIds,
      selectedEvalNames: Array.from(selectedEvalNames),
      trainingRunPolicySelector,
      selectedMetric,
    }
  }

  // Helper function to compare dashboard states
  const isDashboardStateChanged = () => {
    if (!savedDashboard) return true

    const currentState = getDashboardState()
    const originalDashboardState = savedDashboard.dashboard_state
    return (
      JSON.stringify(currentState.selectedTrainingRunIds) !==
        JSON.stringify(originalDashboardState.selectedTrainingRunIds) ||
      JSON.stringify(currentState.selectedRunFreePolicyIds) !==
        JSON.stringify(originalDashboardState.selectedRunFreePolicyIds) ||
      JSON.stringify(currentState.selectedEvalNames) !== JSON.stringify(originalDashboardState.selectedEvalNames) ||
      currentState.trainingRunPolicySelector !== originalDashboardState.trainingRunPolicySelector ||
      currentState.selectedMetric !== originalDashboardState.selectedMetric
    )
  }

  const restoreDashboardState = async (state: DashboardState) => {
    setSelectedTrainingRunIds(state.selectedTrainingRunIds || [])
    setSelectedRunFreePolicyIds(state.selectedRunFreePolicyIds || [])
    setSelectedEvalNames(new Set(state.selectedEvalNames || []))
    setTrainingRunPolicySelector(state.trainingRunPolicySelector || 'latest')
    setSelectedMetric(state.selectedMetric || '')

    await generateScorecard(
      state.selectedTrainingRunIds,
      state.selectedRunFreePolicyIds,
      new Set(state.selectedEvalNames),
      state.selectedMetric
    )
  }

  const handleCreateDashboard = async (dashboardData: SavedDashboardCreate) => {
    try {
      const dashboardState = getDashboardState()
      const saveData = {
        ...dashboardData,
        dashboard_state: dashboardState,
      }

      const savedDashboard = await repo.createSavedDashboard(saveData)

      // Update URL to include the saved dashboard ID
      const newSearchParams = new URLSearchParams(searchParams)
      newSearchParams.set('saved_id', savedDashboard.id)
      setSearchParams(newSearchParams)

      setShowSaveModal(false)
    } catch (error) {
      console.error('Failed to save dashboard:', error)
      throw error
    }
  }

  const loadSavedDashboard = async (savedId: string) => {
    try {
      const savedDashboard = await repo.getSavedDashboard(savedId)
      if (savedDashboard.dashboard_state) {
        setSavedDashboard(savedDashboard)
        await restoreDashboardState(savedDashboard.dashboard_state)
      }
    } catch (error) {
      console.error('Failed to load saved dashboard:', error)
    }
  }

  const savedId = searchParams.get('saved_id')
  const dashboardName = savedDashboard?.name ?? 'Policy Scorecard'

  const handleDashboardButtonClick = async () => {
    if (savedId) {
      await repo.updateDashboardState(savedId, getDashboardState())
      await loadSavedDashboard(savedId)
    } else {
      setShowSaveModal(true)
    }
  }

  // Load saved dashboard on mount if saved_id parameter is present
  useEffect(() => {
    if (savedId) {
      loadSavedDashboard(savedId)
    }
  }, [savedId])

  return (
    <div className={styles.dashboardContainer}>
      <div className={styles.dashboardContent}>
        <div className={styles.dashboardHeader}>
          <h1 className={styles.dashboardTitle}>{dashboardName}</h1>
          <p className={styles.dashboardSubtitle}>
            Select policies and evaluations to generate interactive scorecards for analysis.
          </p>
        </div>

        {error && (
          <div className={styles.errorContainer}>
            <div className={styles.errorTitle}>Error</div>
            <div className={styles.errorMessage}>{error}</div>
          </div>
        )}

        <div className={styles.controlsSection}>
          <div className={styles.controlsHeader}>
            <h2 className={styles.controlsTitle}>Configuration Controls</h2>
            <button
              className={styles.toggleButton}
              onClick={() => setControlsExpanded(!controlsExpanded)}
              aria-expanded={controlsExpanded}
            >
              {controlsExpanded ? '▼' : '▶'} {controlsExpanded ? 'Hide' : 'Show'} Controls
            </button>
          </div>

          {controlsExpanded && (
            <div className={styles.widgetsGrid}>
              {/* Policy Selection */}
              <div className={styles.widget}>
                <h3 className={styles.widgetTitle}>Policy Selection</h3>
                <div className={styles.widgetContent}>
                  {loading.policies ? (
                    <div className={styles.loadingContainer}>
                      <span className={styles.loadingSpinner}></span>
                      Loading policies...
                    </div>
                  ) : (
                    <PolicySelector
                      policies={policies}
                      selectedTrainingRunIds={selectedTrainingRunIds}
                      selectedRunFreePolicyIds={selectedRunFreePolicyIds}
                      onTrainingRunSelectionChange={setSelectedTrainingRunIds}
                      onRunFreePolicySelectionChange={setSelectedRunFreePolicyIds}
                    />
                  )}
                </div>
              </div>

              {/* Evaluation Selection */}
              <div className={styles.widget}>
                <h3 className={styles.widgetTitle}>Evaluation Selection</h3>
                <div className={styles.widgetContent}>
                  <EvalSelector
                    evalNames={evalNames}
                    selectedEvalNames={selectedEvalNames}
                    onSelectionChange={setSelectedEvalNames}
                    loading={loading.evalCategories}
                  />
                </div>
              </div>

              {/* Training Run Policy Selector */}
              <div className={styles.widget}>
                <h3 className={styles.widgetTitle}>Training Run Policy Selector</h3>
                <div className={styles.widgetContent}>
                  <TrainingRunPolicySelector
                    value={trainingRunPolicySelector}
                    onChange={setTrainingRunPolicySelector}
                    disabled={selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0}
                  />
                </div>
              </div>

              {/* Metric Selection */}
              <div className={styles.widget}>
                <h3 className={styles.widgetTitle}>Metric Selection</h3>
                <div className={styles.widgetContent}>
                  <MetricSelector
                    metrics={availableMetrics}
                    selectedMetric={selectedMetric}
                    onSelectionChange={setSelectedMetric}
                    loading={loading.metrics}
                    disabled={
                      (selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0) ||
                      selectedEvalNames.size === 0
                    }
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Generate Scorecard Button */}
        <div className={styles.generateScorecardContainer}>
          <div className={styles.generateScorecardButtonWrapper}>
            <div className={styles.dashboardActions}>
              <button
                onClick={generateScorecardCallback}
                disabled={!canGenerateScorecard}
                className={styles.generateScorecardButton}
              >
                {loading.scorecard ? (
                  <>
                    <span className={styles.loadingSpinner}></span>
                    Generating Scorecard...
                  </>
                ) : (
                  'Generate Scorecard'
                )}
              </button>
              <button
                onClick={handleDashboardButtonClick}
                disabled={!isDashboardStateChanged()}
                className={styles.saveDashboardButton}
              >
                {savedId ? 'Update Dashboard' : 'Save Dashboard'}
              </button>
            </div>
            <div className={styles.buttonHelpText}>
              {selectedTrainingRunIds.length + selectedRunFreePolicyIds.length} policies, {selectedEvalNames.size}{' '}
              evaluations
              {selectedMetric && `, using ${selectedMetric} metric`}
            </div>
            {!canGenerateScorecard && (
              <div className={styles.validationMessage}>
                {selectedTrainingRunIds.length === 0 &&
                  selectedRunFreePolicyIds.length === 0 &&
                  'Please select training runs or policies'}
                {(selectedTrainingRunIds.length > 0 || selectedRunFreePolicyIds.length > 0) &&
                  selectedEvalNames.size === 0 &&
                  'Please select evaluations'}
                {(selectedTrainingRunIds.length > 0 || selectedRunFreePolicyIds.length > 0) &&
                  selectedEvalNames.size > 0 &&
                  !selectedMetric &&
                  'Please select a metric'}
                {Object.values(loading).some(Boolean) && 'Loading...'}
              </div>
            )}
          </div>
        </div>

        {/* Scorecard Display */}
        {scorecardData && (
          <div className={styles.scorecardContainer}>
            <Scorecard
              data={scorecardData}
              selectedMetric={selectedMetric}
              setSelectedCell={setSelectedCell}
              openReplayUrl={openReplayUrl}
              numPoliciesToShow={scorecardData.policyNames.length} // Show all policies
            />

            <MapViewer
              selectedEval={selectedEval}
              isViewLocked={isViewLocked}
              selectedReplayUrl={selectedReplayUrl}
              onToggleLock={toggleLock}
              onReplayClick={handleReplayClick}
            />
          </div>
        )}

        {/* Save Dashboard Modal */}
        <SaveDashboardModal
          isOpen={showSaveModal}
          onClose={() => setShowSaveModal(false)}
          onSave={handleCreateDashboard}
        />
      </div>
    </div>
  )
}
