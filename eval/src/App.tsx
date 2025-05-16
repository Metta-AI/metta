import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import * as duckdb from '@duckdb/duckdb-wasm';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdb_wasm_eh from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import eh_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';

// CSS for map viewer
const MAP_VIEWER_CSS = `
.map-viewer {
    position: relative;
    width: 100%;
    margin-top: 20px;
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
`;

// Types
interface MatrixRow {
  policy_uri: string
  eval_name: string
  value: number
  replay_url?: string
}


// Initialize DuckDB
async function initDuckDB() {
  const MANUAL_BUNDLES: duckdb.DuckDBBundles = {
    mvp: {
        mainModule: duckdb_wasm,
        mainWorker: mvp_worker,
    },
    eh: {
        mainModule: duckdb_wasm_eh,
        mainWorker: eh_worker,
    },
  };
  // Select a bundle based on browser checks
  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);
  // Instantiate the asynchronous version of DuckDB-wasm
  const worker = new Worker(bundle.mainWorker!);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  return db
}


// Load data from DuckDB file
async function loadData(dbUri: string) {
  const db = await initDuckDB()
  
  const conn = await db.connect()
  await conn.send(`ATTACH DATABASE '${dbUri}' AS eval`);
  await conn.send(`USE eval`);
  
  // Query data for heatmap
  const table = await conn.query(`
    WITH potential AS (
      SELECT 
        policy_key,
        policy_version,
        sim_env,
        COUNT(*) AS potential_cnt
      FROM policy_simulation_agent_samples
      GROUP BY policy_key, policy_version, sim_env
    ),
    recorded AS (
      SELECT 
        policy_key,
        policy_version,
        sim_env,
        SUM(value) AS recorded_sum
      FROM policy_simulation_agent_metrics
      WHERE metric = 'reward'  -- TODO: make configurable
      GROUP BY policy_key, policy_version, sim_env
    )
    SELECT
      potential.policy_key || ':v' || potential.policy_version AS policy_uri,
      potential.sim_env AS eval_name,
      COALESCE(recorded.recorded_sum, 0) * 1.0 / potential.potential_cnt AS value
    FROM potential
    LEFT JOIN recorded
      USING (policy_key, policy_version, sim_env)
    ORDER BY policy_uri, eval_name
  `)

  const matrix = table.toArray();
  
  await conn.close()
  return matrix
}

function App() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [matrix, setMatrix] = useState<MatrixRow[]>([])
  const [selectedEval, setSelectedEval] = useState<string | null>(null)
  const [isViewLocked, setIsViewLocked] = useState(false)
  const [isMouseOverMap, setIsMouseOverMap] = useState(false)
  const [isMouseOverHeatmap, setIsMouseOverHeatmap] = useState(false)
  
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
    const dbUri = import.meta.env.VITE_EVAL_DB_URI
    if (!dbUri) {
      setError('VITE_EVAL_DB_URI environment variable not set')
      setLoading(false)
      return
    }
    
    loadData(dbUri)
      .then(data => {
        setMatrix(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])
  
  if (loading) {
    return <div>Loading data...</div>
  }
  
  if (error) {
    return <div style={{ color: 'red' }}>Error: {error}</div>
  }
  
  // Convert to heatmap format
  const policies = [...new Set(matrix.map(r => r.policy_uri))]
  const envs = [...new Set(matrix.map(r => r.eval_name))]
  const shortNames = envs.map(getShortName);
  const z = policies.map(policy => 
    envs.map(env => {
      const row = matrix.find(r => r.policy_uri === policy && r.eval_name === env)
      return row ? row.value : 0
    })
  )

  // Map viewer functions
  const handleHeatmapHover = (event: any) => {
    if (event.points && event.points[0]) {
      const evalName = event.points[0].x;
      const policyUri = event.points[0].y;
      const row = matrix.find(r => r.policy_uri === policyUri && r.eval_name === evalName);
      if (!isViewLocked && evalName.toLowerCase() !== 'overall') {
        setSelectedEval(evalName);
      }
    }
  };

  const handleHeatmapLeave = () => {
    setIsMouseOverHeatmap(false);
    if (!isViewLocked && !isMouseOverMap) {
      setTimeout(() => {
        if (!isMouseOverHeatmap && !isMouseOverMap) {
          setSelectedEval(null);
        }
      }, 100);
    }
  };

  const handleHeatmapEnter = () => {
    setIsMouseOverHeatmap(true);
  };

  const toggleLock = () => {
    setIsViewLocked(!isViewLocked);
  };

  const handleMapEnter = () => {
    setIsMouseOverMap(true);
  };

  const handleMapLeave = () => {
    setIsMouseOverMap(false);
    if (!isViewLocked) {
      setTimeout(() => {
        if (!isMouseOverHeatmap && !isMouseOverMap) {
          setSelectedEval(null);
        }
      }, 100);
    }
  };

  const handleReplayClick = () => {
    if (selectedEval) {
      const row = matrix.find(r => r.eval_name === selectedEval);
      if (row?.replay_url) {
        window.open(row.replay_url, '_blank');
      }
    }
  };
  
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
          paddingBottom: '10px'
        }}>
          Policy Evaluation Dashboard
        </h1>
        
        <div onMouseEnter={handleHeatmapEnter} onMouseLeave={handleHeatmapLeave}>
          <Plot
            data={[{
              z,
              x: shortNames,
              y: policies,
              type: 'heatmap',
              colorscale: 'Viridis',
              colorbar: {
                title: 'reward'  // TODO: make configurable
              }
            }]}
            layout={{
              title: 'Policy Evaluation Report: reward',  // TODO: make configurable
              height: 600,
              width: 900,
              margin: { t: 50, b: 100, l: 100, r: 50 }
            }}
            onHover={handleHeatmapHover}
          />
        </div>

        {/* Map Viewer */}
        <div 
          className="map-viewer" 
          onMouseEnter={handleMapEnter}
          onMouseLeave={handleMapLeave}
        >
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
              className={`map-button ${!selectedEval ? 'disabled' : ''}`}
              onClick={handleReplayClick}
              title="Open replay in Mettascope"
              disabled={!selectedEval}
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