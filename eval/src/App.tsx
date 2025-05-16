import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import * as duckdb from '@duckdb/duckdb-wasm';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdb_wasm_eh from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import eh_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';



// Types
interface MatrixRow {
  policy_uri: string
  eval_name: string
  value: number
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
  const z = envs.map(env => 
    policies.map(policy => {
      const row = matrix.find(r => r.policy_uri === policy && r.eval_name === env)
      return row ? row.value : 0
    })
  )
  
  return (
    <div style={{ 
      fontFamily: 'Arial, sans-serif',
      margin: 0,
      padding: '20px',
      background: '#f8f9fa'
    }}>
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
        
        <Plot
          data={[{
            z,
            x: policies,
            y: envs,
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
        />
      </div>
    </div>
  )
}

export default App 