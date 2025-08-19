import { useState, useEffect } from 'react'
import { fetchAvailableSweeps } from '../api/sweepApi'

interface SweepSelectorProps {
  onSelectSweep: (sweepName: string) => void
}

export function SweepSelector({ onSelectSweep }: SweepSelectorProps) {
  const [sweeps, setSweeps] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [customSweep, setCustomSweep] = useState('')

  useEffect(() => {
    loadSweeps()
  }, [])

  const loadSweeps = async () => {
    try {
      const data = await fetchAvailableSweeps()
      setSweeps(data)
      setError(null)
    } catch (err) {
      // Don't show error if backend is just not running yet
      console.log('Could not fetch sweeps - backend may not be running')
      setSweeps([])
    } finally {
      setLoading(false)
    }
  }

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (customSweep.trim()) {
      onSelectSweep(customSweep.trim())
    }
  }

  return (
    <div className="card">
      <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '20px' }}>
        Select a Sweep to Analyze
      </h2>

      {loading && (
        <div className="alert alert-primary">
          <div className="spinner spinner-sm" />
          Loading available sweeps...
        </div>
      )}

      {error && (
        <div className="alert alert-danger">
          Error: {error}
        </div>
      )}

      {!loading && !error && (
        <>
          {sweeps.length > 0 && (
            <div style={{ marginBottom: '30px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 500, marginBottom: '12px' }}>
                Recent Sweeps
              </h3>
              <div style={{ display: 'grid', gap: '12px', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))' }}>
                {sweeps.map(sweep => (
                  <button
                    key={sweep}
                    className="card"
                    onClick={() => onSelectSweep(sweep)}
                    style={{
                      textAlign: 'left',
                      cursor: 'pointer',
                      border: '1px solid #dee2e6',
                      transition: 'all 0.2s',
                      ':hover': {
                        borderColor: '#007bff',
                        boxShadow: '0 2px 8px rgba(0, 123, 255, 0.15)'
                      }
                    }}
                  >
                    <div style={{ fontWeight: 500, marginBottom: '4px' }}>{sweep}</div>
                    <div style={{ fontSize: '12px', color: '#6c757d' }}>
                      Click to view analysis
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 500, marginBottom: '12px' }}>
              Enter Sweep Name Manually
            </h3>
            <form onSubmit={handleCustomSubmit} style={{ display: 'flex', gap: '12px' }}>
              <input
                type="text"
                className="form-control"
                placeholder="e.g., my-sweep-2024"
                value={customSweep}
                onChange={(e) => setCustomSweep(e.target.value)}
                style={{ flex: 1 }}
              />
              <button type="submit" className="btn btn-primary">
                Load Sweep
              </button>
            </form>
          </div>
        </>
      )}
    </div>
  )
}