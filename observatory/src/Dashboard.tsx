import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { HeatmapData, PolicySelector, Repo, SavedDashboard, SavedDashboardCreate } from './repo'
import { MapViewer } from './MapViewer'
import { Heatmap } from './Heatmap'
import { SaveDashboardModal } from './SaveDashboardModal'
import { MultiSelectDropdown } from './MultiSelectDropdown'
import { SuiteTabs } from './SuiteTabs'

// CSS for dashboard
const DASHBOARD_CSS = `

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-primary {
  background: #007bff;
  color: #fff;
}

.btn-primary:hover {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover {
  background: #545b62;
}

/* Policy selector styles */
.policy-selector {
  margin: 20px 0;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.policy-selector-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.policy-selector-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.policy-selector-controls {
  display: flex;
  gap: 10px;
}

.policy-selector-btn {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: #fff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s ease;
}

.policy-selector-btn:hover {
  background: #f8f8f8;
}

.policy-selector-btn.active {
  background: #007bff;
  color: #fff;
  border-color: #007bff;
}
`

interface DashboardProps {
  repo: Repo
}

export function Dashboard({ repo }: DashboardProps) {
  // Data state
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null)
  const [metrics, setMetrics] = useState<string[]>([])
  const [suites, setSuites] = useState<string[]>([])

  // UI state
  const [selectedMetric, setSelectedMetric] = useState<string>('reward')
  const [selectedSuite, setSelectedSuite] = useState<string>('navigation')
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [selectedCell, setSelectedCell] = useState<{
    policyUri: string
    evalName: string
  } | null>(null)
  const [numPoliciesToShow, setNumPoliciesToShow] = useState(20)
  const [selectedPolicies, setSelectedPolicies] = useState<Set<string>>(new Set())
  const [policySelector, setPolicySelector] = useState<PolicySelector>('latest')

  // Save dashboard state
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [savedId, setSavedId] = useState<string | null>(null)
  const [savedDashboard, setSavedDashboard] = useState<SavedDashboard | null>(null)

  const location = useLocation()

  // Initialize data and load saved dashboard if provided
  useEffect(() => {
    const initializeData = async () => {
      const urlParams = new URLSearchParams(location.search)
      const savedIdParam = urlParams.get('saved_id')

      // Load suites first
      const suitesData = await repo.getSuites()
      setSuites(suitesData)

      if (savedIdParam) {
        // Load saved dashboard
        try {
          const dashboard = await repo.getSavedDashboard(savedIdParam)
          const state = dashboard.dashboard_state
          setSelectedSuite(state.suite || suitesData[0])
          setSelectedMetric(state.metric || 'reward')
          setNumPoliciesToShow(state.num_policies_to_show || 20)
          setSelectedPolicies(new Set(state.selected_policies || []))
          setPolicySelector(state.policy_selector || 'latest')
          setSavedId(savedIdParam)
          setSavedDashboard(dashboard)
        } catch (err) {
          console.error('Failed to load shared dashboard:', err)
          // Fallback to first suite if saved dashboard fails
          setSelectedSuite(suitesData[0])
        }
      } else {
        // No saved dashboard, use first suite
        setSelectedSuite(suitesData[0])
      }
    }

    initializeData()
  }, [location.search, repo])

  // Load metrics when suite changes
  useEffect(() => {
    const loadSuiteData = async () => {
      if (!selectedSuite) return

      const metricsData = await repo.getAllMetrics()
      setMetrics(metricsData)
    }

    loadSuiteData()
  }, [selectedSuite, repo])

  // Load heatmap data when suite, metric, group metric, or policy selector changes
  useEffect(() => {
    const loadHeatmapData = async () => {
      if (!selectedSuite || !selectedMetric) return

      const heatmapData = await repo.getHeatmapData(
        selectedMetric,
        selectedSuite,
        policySelector
      )
      setHeatmapData(heatmapData)
    }

    loadHeatmapData()
  }, [selectedSuite, selectedMetric, policySelector, repo])

  const handleSaveDashboard = async (dashboardData: SavedDashboardCreate) => {
    try {
      const fullDashboardData: SavedDashboardCreate = {
        ...dashboardData,
        dashboard_state: {
          suite: selectedSuite,
          metric: selectedMetric,
          num_policies_to_show: numPoliciesToShow,
          selected_policies: Array.from(selectedPolicies),
          policy_selector: policySelector,
        },
      }

      if (savedId) {
        // Update existing dashboard
        const updatedDashboard = await repo.updateSavedDashboard(savedId, fullDashboardData)
        setSavedDashboard(updatedDashboard)
      } else {
        // Create new dashboard
        const newDashboard = await repo.createSavedDashboard(fullDashboardData)
        setSavedId(newDashboard.id)
        setSavedDashboard(newDashboard)
      }
    } catch (err: any) {
      throw new Error(err.message || 'Failed to save dashboard')
    }
  }

  if (!heatmapData) {
    return <div>Loading...</div>
  }

  // Component functions

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

  const selectedCellData = selectedCell ? heatmapData?.cells[selectedCell.policyUri]?.[selectedCell.evalName] : null
  const selectedEval = selectedCellData?.evalName ?? null
  const selectedReplayUrl = selectedCellData?.replayUrl ?? null

  // Policy selection functions
  const selectAllPolicies = () => {
    const allPolicies = Object.keys(heatmapData.cells)
    setSelectedPolicies(new Set(allPolicies))
  }

  const clearPolicySelection = () => {
    setSelectedPolicies(new Set())
  }

  // Filter heatmap data based on selected policies
  const filteredHeatmapData =
    selectedPolicies.size > 0
      ? {
          ...heatmapData,
          cells: Object.fromEntries(
            Object.entries(heatmapData.cells).filter(([policyUri]) => selectedPolicies.has(policyUri))
          ),
          policyAverageScores: Object.fromEntries(
            Object.entries(heatmapData.policyAverageScores).filter(([policyUri]) => selectedPolicies.has(policyUri))
          ),
        }
      : heatmapData

  // Get sorted policies for display
  const sortedPolicies = Object.keys(heatmapData.cells).sort(
    (a, b) => heatmapData.policyAverageScores[b] - heatmapData.policyAverageScores[a]
  )

  // Convert policies to options for MultiSelectDropdown
  const policyOptions = sortedPolicies.map((policyUri) => ({
    value: policyUri,
    label: policyUri,
    metadata: {
      score: heatmapData.policyAverageScores[policyUri],
    },
  }))

  return (
    <div
      style={{
        padding: '20px',
        background: '#f8f9fa',
        minHeight: 'calc(100vh - 60px)',
      }}
    >
      <style>{DASHBOARD_CSS}</style>
      <div
        style={{
          maxWidth: '1200px',
          margin: '0 auto',
          background: '#fff',
          padding: '20px',
          borderRadius: '5px',
          boxShadow: '0 2px 4px rgba(0,0,0,.1)',
        }}
      >
        {savedDashboard && (
          <div
            style={{
              textAlign: 'center',
              marginBottom: '20px',
              paddingBottom: '20px',
              borderBottom: '1px solid #eee',
            }}
          >
            <h1
              style={{
                margin: 0,
                color: '#333',
                fontSize: '24px',
                fontWeight: '600',
              }}
            >
              {savedDashboard.name}
            </h1>
            {savedDashboard.description && (
              <p
                style={{
                  margin: '8px 0 0 0',
                  color: '#666',
                  fontSize: '16px',
                }}
              >
                {savedDashboard.description}
              </p>
            )}
          </div>
        )}

        <SuiteTabs
          suites={suites}
          selectedSuite={selectedSuite}
          onSuiteChange={setSelectedSuite}
          rightContent={
            <button className="btn btn-secondary" onClick={() => setShowSaveModal(true)}>
              {savedId ? 'Update Dashboard' : 'Save Dashboard'}
            </button>
          }
        />

        <SaveDashboardModal
          isOpen={showSaveModal}
          onClose={() => setShowSaveModal(false)}
          onSave={handleSaveDashboard}
          initialName={savedDashboard?.name || ''}
          initialDescription={savedDashboard?.description || ''}
          isUpdate={!!savedId}
        />

        {filteredHeatmapData && (
          <Heatmap
            data={filteredHeatmapData}
            selectedMetric={selectedMetric}
            setSelectedCell={setSelectedCellIfNotLocked}
            openReplayUrl={openReplayUrl}
            numPoliciesToShow={numPoliciesToShow}
          />
        )}

        {/* Controls Section - Two Column Layout */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '20px',
            marginTop: '30px',
            marginBottom: '30px',
          }}
        >
          {/* Left Column - Heatmap Controls */}
          <div
            style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #e9ecef',
            }}
          >
            <h3
              style={{
                margin: '0 0 15px 0',
                fontSize: '16px',
                fontWeight: '600',
                color: '#333',
              }}
            >
              Heatmap Controls
            </h3>

            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '15px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <div style={{ color: '#666', fontSize: '14px', minWidth: '120px' }}>Heatmap Metric</div>
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #ddd',
                    fontSize: '14px',
                    flex: '1',
                    backgroundColor: '#fff',
                    cursor: 'pointer',
                  }}
                >
                  {metrics.map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </select>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <div style={{ color: '#666', fontSize: '14px', minWidth: '120px' }}>Number of policies:</div>
                <input
                  type="number"
                  value={numPoliciesToShow}
                  onChange={(e) => setNumPoliciesToShow(parseInt(e.target.value))}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #ddd',
                    fontSize: '14px',
                    flex: '1',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Right Column - Policy Selection */}
          <div
            style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #e9ecef',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px',
              }}
            >
              <h3
                style={{
                  margin: 0,
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#333',
                }}
              >
                Policy Selection ({selectedPolicies.size} selected)
              </h3>
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                }}
              >
                <button className="policy-selector-btn" onClick={selectAllPolicies}>
                  Select All
                </button>
                <button className="policy-selector-btn" onClick={clearPolicySelection}>
                  Clear All
                </button>
              </div>
            </div>
            <MultiSelectDropdown
              options={policyOptions}
              selectedValues={selectedPolicies}
              onSelectionChange={setSelectedPolicies}
              placeholder="Select policies"
              searchPlaceholder="Search policies..."
              width="100%"
            />

            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '15px',
                marginTop: '15px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <div style={{ color: '#666', fontSize: '14px', minWidth: '120px' }}>Training Run Policy Selection</div>
                <select
                  value={policySelector}
                  onChange={(e) => setPolicySelector(e.target.value as PolicySelector)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '4px',
                    border: '1px solid #ddd',
                    fontSize: '14px',
                    flex: '1',
                    backgroundColor: '#fff',
                    cursor: 'pointer',
                  }}
                >
                  <option value="latest">Latest</option>
                  <option value="best">Best</option>
                </select>
              </div>
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
