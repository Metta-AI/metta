import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Repo, Leaderboard } from './repo'

// CSS for leaderboards
const LEADERBOARDS_CSS = `
.leaderboard-list {
  max-width: 800px;
  margin: 0 auto;
}

.leaderboard-item {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
  transition: all 0.2s ease;
}

.leaderboard-item:hover {
  box-shadow: 0 4px 8px rgba(0,0,0,.15);
  transform: translateY(-2px);
}

.leaderboard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.leaderboard-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.leaderboard-actions {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 6px 12px;
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

.btn-danger {
  background: #dc3545;
  color: #fff;
}

.btn-danger:hover {
  background: #c82333;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover {
  background: #5a6268;
}

.btn-success {
  background: #28a745;
  color: #fff;
}

.btn-success:hover {
  background: #218838;
}

.leaderboard-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.meta-item {
  display: flex;
  flex-direction: column;
}

.meta-label {
  font-size: 12px;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.meta-value {
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

.leaderboard-dates {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #999;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #666;
}

.empty-state h3 {
  margin-bottom: 12px;
  color: #333;
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

.create-leaderboard-section {
  text-align: center;
  margin-bottom: 30px;
}

.create-leaderboard-section h2 {
  color: #333;
  margin-bottom: 20px;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
}

.tag {
  background: #e9ecef;
  color: #495057;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
`

interface LeaderboardsProps {
  repo: Repo
  currentUser: string
}

export function Leaderboards({ repo }: LeaderboardsProps) {
  const [leaderboards, setLeaderboards] = useState<Leaderboard[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadLeaderboards()
  }, [])

  const loadLeaderboards = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await repo.listLeaderboards()
      setLeaderboards(response.leaderboards)
    } catch (err: any) {
      setError(err.message || 'Failed to load leaderboards')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (leaderboardId: string) => {
    if (!confirm('Are you sure you want to delete this leaderboard?')) {
      return
    }

    try {
      await repo.deleteLeaderboard(leaderboardId)
      setLeaderboards(leaderboards.filter((l) => l.id !== leaderboardId))
    } catch (err: any) {
      alert(`Failed to delete leaderboard: ${err.message}`)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString()
  }

  const handleCreateNew = () => {
    navigate('/leaderboards/create')
  }

  const handleViewLeaderboard = (leaderboard: Leaderboard) => {
    navigate(`/leaderboards/${leaderboard.id}`)
  }

  const handleEditLeaderboard = (leaderboard: Leaderboard) => {
    navigate(`/leaderboards/${leaderboard.id}/edit`)
  }

  if (loading) {
    return (
      <div style={{ padding: '20px' }}>
        <style>{LEADERBOARDS_CSS}</style>
        <div className="loading">
          <h3>Loading leaderboards...</h3>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '20px' }}>
        <style>{LEADERBOARDS_CSS}</style>
        <div className="error">
          <h3>Error loading leaderboards</h3>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={loadLeaderboards}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      style={{
        padding: '20px',
        background: '#f8f9fa',
        minHeight: 'calc(100vh - 60px)',
      }}
    >
      <style>{LEADERBOARDS_CSS}</style>

      <div className="leaderboard-list">
        <h1 style={{ marginBottom: '30px', color: '#333', textAlign: 'center' }}>Leaderboards</h1>

        <div className="create-leaderboard-section">
          <button className="btn btn-success" onClick={handleCreateNew}>
            + Create Leaderboard
          </button>
        </div>

        {leaderboards.length === 0 ? (
          <div className="empty-state">
            <h3>No leaderboards created</h3>
            <p>You haven't created any leaderboards yet. Create one to start tracking policy performance.</p>
          </div>
        ) : (
          leaderboards.map((leaderboard) => (
            <div key={leaderboard.id} className="leaderboard-item">
              <div className="leaderboard-header">
                <h3 className="leaderboard-title">{leaderboard.name}</h3>
                <div className="leaderboard-actions">
                  <button className="btn btn-primary" onClick={() => handleViewLeaderboard(leaderboard)}>
                    View
                  </button>
                  <button className="btn btn-secondary" onClick={() => handleEditLeaderboard(leaderboard)}>
                    Edit
                  </button>
                  <button className="btn btn-danger" onClick={() => handleDelete(leaderboard.id)}>
                    Delete
                  </button>
                </div>
              </div>

              <div className="leaderboard-meta">
                <div className="meta-item">
                  <div className="meta-label">Metric</div>
                  <div className="meta-value">{leaderboard.metric}</div>
                </div>
                <div className="meta-item">
                  <div className="meta-label">Start Date Filter</div>
                  <div className="meta-value">{leaderboard.start_date}</div>
                </div>
                <div className="meta-item">
                  <div className="meta-label">Created</div>
                  <div className="meta-value">{formatDate(leaderboard.created_at)}</div>
                </div>
              </div>

              <div className="meta-item">
                <div className="meta-label">Evaluations</div>
                <div className="tag-list">
                  {leaderboard.evals.map((evalName, index) => (
                    <span key={index} className="tag">
                      {evalName}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
