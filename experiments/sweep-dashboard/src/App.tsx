import { useState, useEffect } from 'react'
import { SweepDashboard } from './components/SweepDashboard'
import { SweepSelector } from './components/SweepSelector'
import { SweepData } from './types'
import { fetchSweepData } from './api/sweepApi'

function App() {
  const [selectedSweep, setSelectedSweep] = useState<string | null>(null)
  const [sweepData, setSweepData] = useState<SweepData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (selectedSweep) {
      loadSweepData(selectedSweep)
    }
  }, [selectedSweep])

  const loadSweepData = async (sweepName: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchSweepData(sweepName)
      setSweepData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sweep data')
      setSweepData(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header style={{
        background: 'white',
        borderBottom: '1px solid #dee2e6',
        padding: '16px 0',
        marginBottom: '24px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
      }}>
        <div className="container">
          <h1 style={{ fontSize: '24px', fontWeight: 600, color: '#212529' }}>
            Sweep Analysis Dashboard
          </h1>
        </div>
      </header>

      <main className="container">
        {!selectedSweep ? (
          <SweepSelector onSelectSweep={setSelectedSweep} />
        ) : (
          <>
            <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <button 
                className="btn btn-secondary btn-sm"
                onClick={() => {
                  setSelectedSweep(null)
                  setSweepData(null)
                }}
              >
                ← Back to Sweep List
              </button>
              <h2 style={{ fontSize: '20px', fontWeight: 500, color: '#495057' }}>
                {selectedSweep}
              </h2>
            </div>

            {loading && (
              <div className="alert alert-primary">
                <div className="spinner spinner-sm" />
                Loading sweep data...
              </div>
            )}

            {error && (
              <div className="alert alert-danger">
                Error: {error}
              </div>
            )}

            {sweepData && !loading && (
              <SweepDashboard sweepData={sweepData} sweepName={selectedSweep} />
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App