import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import styles from './Dashboard.module.css'
import { Heatmap } from './Heatmap'
import { MapViewer } from './MapViewer'
import { SaveDashboardModal } from './SaveDashboardModal'
import { EvalSelector } from './components/EvalSelector'
import { MetricSelector } from './components/MetricSelector'
import { PolicySelector } from './components/PolicySelector'
import { SearchInput } from './components/SearchInput'
import { TrainingRunPolicySelector } from './components/TrainingRunPolicySelector'
import type { PolicyHeatmapData, Repo, SavedDashboardCreate } from './repo'

interface DashboardProps {
  repo: Repo
}

// Dashboard state interface for saving/loading
export interface DashboardState {
  selectedTrainingRunIds: Array<string>
  selectedRunFreePolicyIds: Array<string>
  selectedEvalNames: Array<string>
  trainingRunPolicySelector: 'latest' | 'best'
  selectedMetric: string
}

export function Dashboard({ repo }: DashboardProps) {
  const [searchParams, setSearchParams] = useSearchParams()
  // Data state
  const [evalNames, setEvalNames] = useState<Set<string>>(new Set())
  const [availableMetrics, setAvailableMetrics] = useState<Array<string>>([])
  const [heatmapData, setHeatmapData] = useState<PolicyHeatmapData | null>(null)

  // Selection state
  const [selectedTrainingRunIds, setSelectedTrainingRunIds] = useState<Array<string>>([])
  const [selectedRunFreePolicyIds, setSelectedRunFreePolicyIds] = useState<Array<string>>([])
  const [selectedEvalNames, setSelectedEvalNames] = useState<Set<string>>(new Set())
  const [trainingRunPolicySelector, setTrainingRunPolicySelector] = useState<'latest' | 'best'>('latest')
  const [selectedMetric, setSelectedMetric] = useState<string>('')

  // Pagination state
  const [currentPage, setCurrentPage] = useState<number>(1)

  // UI state
  const [policySearchText, setPolicySearchText] = useState<string>('')
  const [loading, setLoading] = useState({
    evalCategories: false,
    metrics: false,
    heatmap: false,
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
  }, [repo, selectedTrainingRunIds, selectedRunFreePolicyIds])

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
      } catch (err) {
        setError(`Failed to load metrics: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setAvailableMetrics([])
        setSelectedMetric('')
      } finally {
        setLoading((prev) => ({ ...prev, metrics: false }))
      }
    }

    loadMetrics()
  }, [repo, selectedTrainingRunIds, selectedRunFreePolicyIds, selectedEvalNames, selectedMetric])

  // Generate heatmap
  const generateHeatmap = async (
    selectedTrainingRunIds: Array<string>,
    selectedRunFreePolicyIds: Array<string>,
    selectedEvalNames: Set<string>,
    selectedMetric: string
  ) => {
    if (
      (selectedTrainingRunIds.length === 0 && selectedRunFreePolicyIds.length === 0) ||
      selectedEvalNames.size === 0 ||
      !selectedMetric
    ) {
      setError('Please select training runs/policies, evaluations, and a metric before generating the heatmap.')
      return
    }

    try {
      setLoading((prev) => ({ ...prev, heatmap: true }))
      setError(null)
      const heatmapResult = await repo.generatePolicyHeatmap({
        training_run_ids: selectedTrainingRunIds,
        run_free_policy_ids: selectedRunFreePolicyIds,
        eval_names: Array.from(selectedEvalNames),
        training_run_policy_selector: trainingRunPolicySelector,
        metric: selectedMetric,
      })
      setHeatmapData(heatmapResult)
      setControlsExpanded(false)
    } catch (err) {
      setError(`Failed to generate heatmap: ${err instanceof Error ? err.message : 'Unknown error'}`)
      setHeatmapData(null)
    } finally {
      setLoading((prev) => ({ ...prev, heatmap: false }))
    }
  }

  const generateHeatmapCallback = async () => {
    await generateHeatmap(selectedTrainingRunIds, selectedRunFreePolicyIds, selectedEvalNames, selectedMetric)
  }

  // Stable handlers for PolicySelector to prevent unnecessary re-renders
  const handleSearchChange = useCallback((searchText: string) => {
    setPolicySearchText(searchText)
  }, [])

  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page)
  }, [])

  const canGenerateHeatmap =
    (selectedTrainingRunIds.length > 0 || selectedRunFreePolicyIds.length > 0) &&
    selectedEvalNames.size > 0 &&
    selectedMetric !== '' &&
    !Object.values(loading).some(Boolean)

  const openReplayUrl = (policyName: string, evalName: string) => {
    const cell = heatmapData?.cells[policyName]?.[evalName]
    if (!cell?.replayUrl) {
      return
    }

    const replay_url_prefix = 'https://metta-ai.github.io/metta/?replayUrl='
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

  const selectedCellData = selectedCell ? heatmapData?.cells[selectedCell.policyUri]?.[selectedCell.evalName] : null
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

  const restoreDashboardState = async (state: DashboardState) => {
    setSelectedTrainingRunIds(state.selectedTrainingRunIds || [])
    setSelectedRunFreePolicyIds(state.selectedRunFreePolicyIds || [])
    setSelectedEvalNames(new Set(state.selectedEvalNames || []))
    setTrainingRunPolicySelector(state.trainingRunPolicySelector || 'latest')
    setSelectedMetric(state.selectedMetric || '')

    await generateHeatmap(
      state.selectedTrainingRunIds,
      state.selectedRunFreePolicyIds,
      new Set(state.selectedEvalNames),
      state.selectedMetric
    )
  }

  const handleSaveDashboard = async (dashboardData: SavedDashboardCreate) => {
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

  const savedId = searchParams.get('saved_id')
  // Load saved dashboard on mount if saved_id parameter is present
  useEffect(() => {
    if (savedId) {
      const loadSavedDashboard = async () => {
        try {
          const savedDashboard = await repo.getSavedDashboard(savedId)
          if (savedDashboard.dashboard_state) {
            await restoreDashboardState(savedDashboard.dashboard_state as DashboardState)
          }
        } catch (error) {
          console.error('Failed to load saved dashboard:', error)
        }
      }

      loadSavedDashboard()
    }
  }, [savedId, repo])

  return (
    <div className={styles.dashboardContainer}>
      <div className={styles.dashboardContent}>
        <div className={styles.dashboardHeader}>
          <h1 className={styles.dashboardTitle}>Policy Heatmap Dashboard</h1>
          <p className={styles.dashboardSubtitle}>
            Select policies and evaluations to generate interactive heatmaps for analysis.
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
                  <SearchInput searchText={policySearchText} onSearchChange={handleSearchChange} disabled={false} />
                  <PolicySelector
                    repo={repo}
                    searchText={policySearchText}
                    selectedTrainingRunIds={selectedTrainingRunIds}
                    selectedRunFreePolicyIds={selectedRunFreePolicyIds}
                    onTrainingRunSelectionChange={setSelectedTrainingRunIds}
                    onRunFreePolicySelectionChange={setSelectedRunFreePolicyIds}
                    currentPage={currentPage}
                    onPageChange={handlePageChange}
                  />
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

        {/* Generate Heatmap Button */}
        <div className={styles.generateHeatmapContainer}>
          <div className={styles.generateHeatmapButtonWrapper}>
            <div className={styles.dashboardActions}>
              <button
                onClick={generateHeatmapCallback}
                disabled={!canGenerateHeatmap}
                className={styles.generateHeatmapButton}
              >
                {loading.heatmap ? (
                  <>
                    <span className={styles.loadingSpinner} />
                    Generating Heatmap...
                  </>
                ) : (
                  'Generate Heatmap'
                )}
              </button>
              <button
                onClick={() => setShowSaveModal(true)}
                disabled={!heatmapData}
                className={styles.saveDashboardButton}
                title={
                  heatmapData
                    ? 'Save current dashboard configuration'
                    : 'Generate a heatmap first to save the dashboard'
                }
              >
                Save Dashboard
              </button>
            </div>
            <div className={styles.buttonHelpText}>
              {selectedTrainingRunIds.length + selectedRunFreePolicyIds.length} policies, {selectedEvalNames.size}{' '}
              evaluations
              {selectedMetric && `, using ${selectedMetric} metric`}
            </div>
            {!canGenerateHeatmap && (
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

        {/* Heatmap Display */}
        {heatmapData && (
          <div className={styles.heatmapContainer}>
            <Heatmap
              data={heatmapData}
              selectedMetric={selectedMetric}
              setSelectedCell={setSelectedCell}
              openReplayUrl={openReplayUrl}
              numPoliciesToShow={heatmapData.policyNames.length} // Show all policies
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
          onSave={handleSaveDashboard}
        />
      </div>
    </div>
  )
}
