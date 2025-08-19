import { useState, useEffect } from 'react'
import { fetchAvailableSweeps, SweepsResponse } from '../api/sweepApi'

interface SweepSelectorProps {
  onSelectSweep: (sweepName: string) => void
}

export function SweepSelector({ onSelectSweep }: SweepSelectorProps) {
  const [sweepsData, setSweepsData] = useState<SweepsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [customSweep, setCustomSweep] = useState('')

  useEffect(() => {
    loadSweeps()
  }, [])

  const loadSweeps = async () => {
    try {
      const data = await fetchAvailableSweeps()
      setSweepsData(data)
      if (data.error && data.sweeps.length === 0) {
        setError(data.error)
      } else {
        setError(null)
      }
    } catch (err) {
      // Don't show error if backend is just not running yet
      console.log('Could not fetch sweeps - backend may not be running')
      setSweepsData({ sweeps: [], count: 0 })
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

      {!loading && !error && sweepsData && (
        <>
          {sweepsData.sweeps.length > 0 && (
            <div style={{ marginBottom: '30px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 500 }}>
                  Available Sweeps ({sweepsData.count})
                </h3>
                {sweepsData.entity && sweepsData.project && (
                  <span style={{ fontSize: '12px', color: '#6c757d' }}>
                    {sweepsData.entity}/{sweepsData.project}
                  </span>
                )}
              </div>
              <div style={{ display: 'grid', gap: '12px', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
                {sweepsData.sweeps.map(sweep => (
                  <button
                    key={sweep}
                    className="card"
                    onClick={() => onSelectSweep(sweep)}
                    style={{
                      textAlign: 'left',
                      cursor: 'pointer',
                      border: '2px solid #dee2e6',
                      transition: 'all 0.2s',
                      padding: '16px',
                      backgroundColor: 'white'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#007bff'
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 123, 255, 0.15)'
                      e.currentTarget.style.transform = 'translateY(-2px)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = '#dee2e6'
                      e.currentTarget.style.boxShadow = 'none'
                      e.currentTarget.style.transform = 'translateY(0)'
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: '4px', color: '#212529' }}>
                      {sweep}
                    </div>
                    <div style={{ fontSize: '12px', color: '#6c757d' }}>
                      Click to view analysis →
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