import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Repo, Leaderboard, ScorecardData, LeaderboardScorecardRequest } from './repo'
import { Scorecard } from './Scorecard'
import { MapViewer } from './MapViewer'
import { METTASCOPE_REPLAY_URL_PREFIX } from './constants'

// CSS for leaderboard view
const LEADERBOARD_VIEW_CSS = `
.view-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.view-header {
  text-align: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.view-title {
  color: #333;
  margin-bottom: 10px;
}

.view-subtitle {
  color: #666;
  font-size: 14px;
}

.controls-section {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
}

.controls-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin: 0 0 16px 0;
}

.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  align-items: end;
}

.control-group {
  display: flex;
  flex-direction: column;
}

.control-label {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin-bottom: 6px;
}

.control-input, .control-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  background: #fff;
}

.control-input:focus, .control-select:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0,123,255,.25);
}


.actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
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
  background: #5a6268;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error {
  text-align: center;
  padding: 40px;
  color: #dc3545;
}
`

interface LeaderboardViewProps {
  repo: Repo
}

export function LeaderboardView({ repo }: LeaderboardViewProps) {
  const { leaderboardId } = useParams<{ leaderboardId: string }>()
  const navigate = useNavigate()
  const [leaderboard, setLeaderboard] = useState<Leaderboard | null>(null)
  const [scorecardData, setScorecardData] = useState<ScorecardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [scorecardLoading, setScorecardLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedCell, setSelectedCell] = useState<{ policyUri: string; evalName: string } | null>(null)
  const [isViewLocked, setIsViewLocked] = useState(false)

  // Controls for scorecard generation
  const [policySelector, setPolicySelector] = useState<'latest' | 'best'>('latest')
  const [numPolicies, setNumPolicies] = useState(20)

  useEffect(() => {
    if (leaderboardId) {
      loadLeaderboard()
    }
  }, [leaderboardId])

  useEffect(() => {
    if (leaderboard) {
      loadScorecardData()
    }
  }, [leaderboard, policySelector, numPolicies])

  // Auto-refresh when leaderboard is building (latest_episode is 0)
  useEffect(() => {
    if (!leaderboard || leaderboard.latest_episode > 0) {
      return
    }

    const interval = setInterval(() => {
      loadLeaderboard()
    }, 5000) // Refresh every 5 seconds

    return () => clearInterval(interval)
  }, [leaderboard, leaderboardId])

  const loadLeaderboard = async () => {
    if (!leaderboardId) return

    try {
      setLoading(true)
      setError(null)
      const data = await repo.getLeaderboard(leaderboardId)
      setLeaderboard(data)
    } catch (err: any) {
      setError(`Failed to load leaderboard: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const loadScorecardData = async () => {
    if (!leaderboardId) return

    try {
      setScorecardLoading(true)
      setError(null)
      const request: LeaderboardScorecardRequest = {
        selector: policySelector,
        num_policies: numPolicies,
      }
      const data = await repo.generateLeaderboardScorecard(leaderboardId, request)
      setScorecardData(data)
    } catch (err: any) {
      setError(`Failed to load scorecard data: ${err.message}`)
    } finally {
      setScorecardLoading(false)
    }
  }

  const handleBack = () => {
    navigate('/leaderboards')
  }

  const openReplayUrl = (policyName: string, evalName: string) => {
    const cell = scorecardData?.cells[policyName]?.[evalName]
    if (!cell?.replayUrl) return

    window.open(METTASCOPE_REPLAY_URL_PREFIX + cell.replayUrl, '_blank')
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
  const selectedThumbnailUrl = selectedCellData?.thumbnailUrl ?? null

  if (loading) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_VIEW_CSS}</style>
        <div className="loading">
          <h3>Loading leaderboard...</h3>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_VIEW_CSS}</style>
        <div className="error">
          <h3>Error</h3>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={loadLeaderboard}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!leaderboard) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_VIEW_CSS}</style>
        <div className="error">
          <h3>Leaderboard not found</h3>
          <button className="btn btn-primary" onClick={handleBack}>
            Back to Leaderboards
          </button>
        </div>
      </div>
    )
  }

  // Check if leaderboard is still building
  if (leaderboard.latest_episode === 0) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_VIEW_CSS}</style>
        <div className="loading">
          <h3>Leaderboard building</h3>
          <p>This may take a few minutes. The page will automatically refresh...</p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
      <style>{LEADERBOARD_VIEW_CSS}</style>

      <div className="view-container">
        <div className="view-header">
          <h1 className="view-title">{leaderboard.name}</h1>
          <p className="view-subtitle">Leaderboard Scorecard - {leaderboard.metric}</p>
        </div>

        {/* Scorecard Controls */}
        <div className="controls-section">
          <h3 className="controls-title">Scorecard Settings</h3>
          <div className="controls-grid">
            <div className="control-group">
              <label className="control-label" htmlFor="policySelector">
                Policy Selector
              </label>
              <select
                id="policySelector"
                className="control-select"
                value={policySelector}
                onChange={(e) => setPolicySelector(e.target.value as 'latest' | 'best')}
              >
                <option value="latest">Latest</option>
                <option value="best">Best</option>
              </select>
            </div>
            <div className="control-group">
              <label className="control-label" htmlFor="numPolicies">
                Number of Policies
              </label>
              <input
                id="numPolicies"
                type="number"
                className="control-input"
                value={numPolicies}
                onChange={(e) => setNumPolicies(parseInt(e.target.value) || 20)}
                min="1"
                max="100"
              />
            </div>
          </div>
        </div>

        {scorecardLoading && (
          <div className="loading">
            <h3>Loading scorecard...</h3>
          </div>
        )}

        {!scorecardLoading && scorecardData && (
          <div>
            <Scorecard
              data={scorecardData}
              selectedMetric={leaderboard.metric}
              setSelectedCell={setSelectedCell}
              openReplayUrl={openReplayUrl}
              numPoliciesToShow={numPolicies}
            />

            <MapViewer
              selectedEval={selectedEval}
              isViewLocked={isViewLocked}
              selectedReplayUrl={selectedReplayUrl}
              selectedThumbnailUrl={selectedThumbnailUrl}
              onToggleLock={toggleLock}
              onReplayClick={handleReplayClick}
            />
          </div>
        )}

        {/* Actions */}
        <div className="actions">
          <button className="btn btn-secondary" onClick={handleBack}>
            Back to Leaderboards
          </button>
        </div>
      </div>
    </div>
  )
}
