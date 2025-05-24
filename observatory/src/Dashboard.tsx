import { useEffect, useState } from 'react'
import { PolicyEvalMetric } from './data_loader'
import { Repo } from './repo'
import { MapViewer } from './MapViewer'
import { Heatmap } from './Heatmap'

// CSS for map viewer
const MAP_VIEWER_CSS = `
.map-viewer {
    position: relative;
    width: 1000px;
    margin: 20px auto;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: #f9f9f9;
    min-height: 300px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.map-viewer-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
    font-size: 18px;
}
.map-viewer-img {
    max-width: 100%;
    max-height: 350px;
    display: block;
    margin: 0 auto;
}
.map-viewer-placeholder {
    text-align: center;
    color: #666;
    padding: 50px 0;
    font-style: italic;
}
.map-viewer-controls {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    gap: 10px;
}
.map-button {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 14px;
}
.map-button svg {
    width: 14px;
    height: 14px;
}
.map-button.locked {
    background: #f0f0f0;
    border-color: #aaa;
}
.map-button:hover {
    background: #f0f0f0;
}
.map-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Tab styles */
.suite-tabs {
  display: flex;
  gap: 2px;
  padding: 4px;
  border-radius: 8px;
  margin-bottom: 20px;
  overflow-x: auto;
  max-width: 1000px;
  margin: 0 auto 20px auto;
}

.suite-tab {
  padding: 8px 16px;
  border: none;
  background: #fff;
  cursor: pointer;
  font-size: 14px;
  color: #666;
  border-radius: 6px;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.suite-tab:hover {
  background: #f8f8f8;
  color: #333;
}

.suite-tab.active {
  background: #007bff;
  color: #fff;
  font-weight: 500;
}

.suite-tab:first-child {
  margin-left: 0;
}

.suite-tab:last-child {
  margin-right: 0;
}
`;

interface DashboardProps {
  repo: Repo;
}

export function Dashboard({ repo }: DashboardProps) {
  // Data state
  const [matrix, setMatrix] = useState<PolicyEvalMetric[]>([])
  const [metrics, setMetrics] = useState<string[]>([])
  const [suites, setSuites] = useState<string[]>([])

  // UI state
  const [selectedMetric, setSelectedMetric] = useState<string>("reward")
  const [selectedSuite, setSelectedSuite] = useState<string>("navigation")
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [lastHoveredCell, setLastHoveredCell] = useState<{shortName: string, policyUri: string} | null>(null)
  const [selectedCell, setSelectedCell] = useState<{shortName: string, policyUri: string} | null>(null)

  const evalLookupMap = new Map<string, Map<string, { evalName: string, replayUrl: string | null }>>();
  matrix.forEach(row => {
    if (!evalLookupMap.has(row.policy_uri)) {
      evalLookupMap.set(row.policy_uri, new Map())
    }
    evalLookupMap.get(row.policy_uri)?.set(row.eval_name.split('/').pop() || row.eval_name, {
      evalName: row.eval_name,
      replayUrl: row.replay_url
    })
  })

  useEffect(() => {
    const loadData = async () => {
      const [metricsData, suitesData] = await Promise.all([
        repo.getMetrics(),
        repo.getSuites()
      ]);
      setMetrics(metricsData);
      setSuites(suitesData);
      setSelectedSuite(suitesData[0]);
    };

    loadData();
  }, []);

  useEffect(() => {
    const loadData = async () => {
      const [matrixData] = await Promise.all([
        repo.getPolicyEvals(selectedMetric, selectedSuite),
      ]);
      setMatrix(matrixData);
    };

    loadData();
  }, [selectedSuite, selectedMetric]);

  // Map viewer functions
  const handleHeatmapHover = (event: any) => {
    if (!event.points?.[0]) return
    
    const shortName = event.points[0].x
    const policyUri = event.points[0].y
    
    setLastHoveredCell({ shortName, policyUri })
    if (!isViewLocked && !(shortName === "overall")) {
      setSelectedCell({shortName, policyUri})
    }
  };

  const openReplayUrl = (policyUri: string, evalShortName: string) => {
    const policyData = evalLookupMap.get(policyUri);
    if (!policyData) return;
    
    const evalData = policyData.get(evalShortName);
    if (!evalData?.replayUrl) return;
    
    const replay_url_prefix = "https://metta-ai.github.io/metta/?replayUrl=";
    window.open(replay_url_prefix + evalData.replayUrl, '_blank');
  }

  const handleHeatmapDoubleClick = () => {
    if (lastHoveredCell) {
      openReplayUrl(lastHoveredCell.policyUri, lastHoveredCell.shortName)
    }
  };

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked);
  };

  const handleReplayClick = () => {
    if (selectedCell) {
      openReplayUrl(selectedCell.policyUri, selectedCell.shortName)
    }
  };

  const selectedCellData = selectedCell ? evalLookupMap.get(selectedCell.policyUri)?.get(selectedCell.shortName) : null
  const selectedEval = selectedCellData?.evalName ?? null
  const selectedReplayUrl = selectedCellData?.replayUrl ?? null

  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif',
      margin: 0,
      padding: '20px',
      background: '#f8f9fa'
    }}>
      <style>{MAP_VIEWER_CSS}</style>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: '#fff',
        padding: '20px',
        borderRadius: '5px',
        boxShadow: '0 2px 4px rgba(0,0,0,.1)'
      }}>
        <h1 style={{
          color: '#333',
          borderBottom: '1px solid #ddd',
          paddingBottom: '10px',
          marginBottom: '20px'
        }}>
          Policy Evaluation Dashboard
        </h1>

        <div className="suite-tabs">
          <div style={{ fontSize: '18px', marginTop: '5px', marginRight: '10px' }}>Eval Suite:</div>
          {suites.map(suite => (
            <button
              key={suite}
              className={`suite-tab ${selectedSuite === suite ? 'active' : ''}`}
              onClick={() => setSelectedSuite(suite)}
            >
              {suite}
            </button>
          ))}
        </div>
        
        <Heatmap
          matrix={matrix}
          selectedMetric={selectedMetric}
          onHover={handleHeatmapHover}
          onDoubleClick={handleHeatmapDoubleClick}
        />

        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          marginTop: '20px',
          marginBottom: '30px',
          gap: '12px'
        }}>
          <div style={{ color: '#666', fontSize: '14px' }}>Heatmap Metric</div>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '14px',
              minWidth: '200px',
              backgroundColor: '#fff',
              cursor: 'pointer'
            }}
          >
            {metrics.map(metric => (
              <option key={metric} value={metric}>
                {metric}
              </option>
            ))}
          </select>
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