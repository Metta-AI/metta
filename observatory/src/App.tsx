import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { PolicyEvalMetric, loadDataFromUri, loadDataFromFile } from './data_loader'
import { DataRepo, Repo } from './repo'

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

// Load data from API endpoints

function App() {
  // Data loading state
  const [dataUri, setDataUri] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [repo, setRepo] = useState<Repo | null>(null)

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
  
  
  // Map image URL helper
  const getShortName = (evalName: string) => {
    if (evalName === 'Overall') return evalName;
    return evalName.split('/').pop() || evalName;
  };

  const getMapImageUrl = (evalName: string) => {
    if (evalName.toLowerCase() === 'overall') return '';
    const shortName = getShortName(evalName);
    return `https://softmax-public.s3.amazonaws.com/policydash/evals/img/${shortName.toLowerCase()}.png`;
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const dataUri = params.get('data');
    if (dataUri) {
      setDataUri(dataUri);
    }
  }, []);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        if (dataUri) {
          const data = await loadDataFromUri(dataUri);
          setRepo(new DataRepo(data));
        }
        setLoading(false);
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadData();
  }, [dataUri]);

  useEffect(() => {
    const loadData = async () => {
    if (repo) {
        const [metricsData, suitesData] = await Promise.all([
          repo.getMetrics(),
          repo.getSuites()
        ]);
        setMetrics(metricsData);
        setSuites(suitesData);
        setSelectedSuite(suitesData[0]);
      }
    };

    loadData();
  }, [repo]);

  useEffect(() => {
    const loadData = async () => {
      if (repo) {
        const [matrixData] = await Promise.all([
          repo.getPolicyEvals(selectedMetric, selectedSuite),
        ]);
        setMatrix(matrixData);
      }
    };

    loadData();
  }, [selectedSuite, selectedMetric, repo]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setLoading(true);
      setError(null);
      setFileError(null);

      const data = await loadDataFromFile(file);
      setRepo(new DataRepo(data));
      setLoading(false);
    } catch (err: any) {
      setFileError('Invalid file format. Please upload a valid JSON file.');
      setLoading(false);
    }
  };

  if (!dataUri) {
    return (
      <div style={{ 
        fontFamily: 'Arial, sans-serif',
        margin: 0,
        padding: '20px',
        background: '#f8f9fa',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{
          maxWidth: '600px',
          margin: '0 auto',
          background: '#fff',
          padding: '40px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,.1)',
          textAlign: 'center'
        }}>
          <h1 style={{
            color: '#333',
            marginBottom: '20px'
          }}>
            Policy Evaluation Dashboard
          </h1>
          <p style={{ marginBottom: '20px', color: '#666' }}>
            Upload your evaluation data or provide a data URI as a query parameter.
          </p>
          <div style={{ marginBottom: '20px' }}>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              style={{
                padding: '10px',
                border: '2px dashed #ddd',
                borderRadius: '4px',
                width: '100%',
                cursor: 'pointer'
              }}
            />
          </div>
          {fileError && (
            <div style={{ color: 'red', marginTop: '10px' }}>
              {fileError}
            </div>
          )}
          <p style={{ color: '#666', fontSize: '14px' }}>
            Or add <code>?data=YOUR_DATA_URI</code> to the URL
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return <div>Loading data...</div>
  }
  
  if (error) {
    return <div style={{ color: 'red' }}>Error: {error}</div>
  }

  const policyEvalMap = new Map<string, Map<string, number>>();
  for (const row of matrix) {
    if (!policyEvalMap.has(row.policy_uri)) {
      policyEvalMap.set(row.policy_uri, new Map())
    }
    policyEvalMap.get(row.policy_uri)?.set(getShortName(row.eval_name), row.value)
  }

  const evalLookupMap = new Map<string, Map<string, { evalName: string, replayUrl: string | null }>>();
  matrix.forEach(row => {
    if (!evalLookupMap.has(row.policy_uri)) {
      evalLookupMap.set(row.policy_uri, new Map())
    }
    evalLookupMap.get(row.policy_uri)?.set(getShortName(row.eval_name), {
      evalName: row.eval_name,
      replayUrl: row.replay_url
    })
  })
  
  // Convert to heatmap format
  const policies = [...new Set(matrix.map(r => r.policy_uri))]
  const envs = [...new Set(matrix.map(r => r.eval_name))]
  const sortedShortNames = envs.map(getShortName).sort((a, b) => a.localeCompare(b));

  const sortedShortNamesWithOverall = ["overall", ...sortedShortNames];

  // Iterate over the policyEvalMap, and for each policy compute the average value of the evals
  policyEvalMap.forEach((evalMap) => {
    const overallValue = Array.from(evalMap.values()).reduce((sum, value) => sum + value, 0) / envs.length;
    evalMap.set("overall", overallValue);
  });
  const sortedPolicies = policies.sort((a, b) => policyEvalMap.get(a)!.get("overall")! - policyEvalMap.get(b)!.get("overall")!);
  // take last 20 of sorted policies
  const y_labels = sortedPolicies.slice(-20)

  const z = y_labels.map(policy => 
    sortedShortNamesWithOverall.map(shortName => policyEvalMap.get(policy)!.get(shortName) || 0)
  )

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
    if (selectedCell ) {
      openReplayUrl(selectedCell.policyUri, selectedCell.shortName)
    }
  };

  const wandb_url = (policyName: string) => {
    const entity = "metta-research"
    const project = "metta"
    let policyKey = policyName
    if (policyName.includes(":v")) {
      policyKey = policyName.split(":v")[0]
    }
    return `https://wandb.ai/${entity}/${project}/runs/${policyKey}`
  }

  const y_label_texts = y_labels.map(policy => {
    return `<a href="${wandb_url(policy)}" target="_blank">${policy}</a>`
  })

  const data: Plotly.Data = {
    z,
    x: sortedShortNamesWithOverall,
    y: y_labels,
    type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: {
      title: {
        text: selectedMetric
      }
    }
  }

  const selectedCellData = selectedCell ? evalLookupMap.get(selectedCell.policyUri)?.get(selectedCell.shortName) : null
  const selectedEval = selectedCellData?.evalName
  const selectedReplayUrl = selectedCellData?.replayUrl
  
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
        
        <Plot
          data={[data]}
          layout={{
            title: {
              text: `Policy Evaluation Report: ${selectedMetric}`,
              font: {
                size: 24
              }
            },
            height: 600,
            width: 1000,
            margin: { t: 50, b: 150, l: 200, r: 50 },
            xaxis: {
              tickangle: -45
            },
            yaxis: {
              tickangle: 0,
              automargin: true,
              ticktext: y_label_texts,
              tickvals: Array.from({ length: y_labels.length }, (_, i) => i),
              tickmode: 'array'
            }
          }}
          style={{
            margin: '0 auto',
            display: 'block'
          }}
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

        {/* Map Viewer */}
        <div className="map-viewer" >
          <div className="map-viewer-title">
            {selectedEval || 'Map Viewer'}
          </div>
          {!selectedEval ? (
            <div className="map-viewer-placeholder">
              Hover over an evaluation name or cell to see the environment map
            </div>
          ) : (
            <img 
              className="map-viewer-img" 
              src={getMapImageUrl(selectedEval)}
              alt={`Environment map for ${selectedEval}`}
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const placeholder = target.parentElement?.querySelector('.map-viewer-placeholder') as HTMLDivElement;
                if (placeholder) {
                  placeholder.textContent = `No map available for ${selectedEval}`;
                  placeholder.style.display = 'block';
                }
              }}
            />
          )}
          
          <div className="map-viewer-controls">
            <button 
              className={`map-button ${isViewLocked ? 'locked' : ''}`}
              onClick={toggleLock}
              title="Lock current view (or click cell)"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
              </svg>
              <span>{isViewLocked ? 'Unlock View' : 'Lock View'}</span>
            </button>
            <button 
              className={`map-button ${!selectedReplayUrl ? 'disabled' : ''}`}
              onClick={handleReplayClick}
              title="Open replay in Mettascope"
              disabled={!selectedReplayUrl}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 0 014.25 4h5a.75.75 0 010 1.5h-5z" clipRule="evenodd" />
                <path fillRule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5 0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z" clipRule="evenodd" />
              </svg>
              <span>Open Replay</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App 