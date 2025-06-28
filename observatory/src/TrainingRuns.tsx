import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { TrainingRun, Repo } from './repo'

const TRAINING_RUNS_CSS = `
.training-runs-container {
  padding: 20px;
  background: #f8f9fa;
  min-height: calc(100vh - 60px);
}

.training-runs-content {
  max-width: 1200px;
  margin: 0 auto;
  background: #fff;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.training-runs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.training-runs-title {
  margin: 0;
  color: #333;
  font-size: 24px;
  font-weight: 600;
}

.training-runs-count {
  color: #666;
  font-size: 14px;
}

.search-box {
  width: 300px;
  padding: 8px 12px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 4px;
  outline: none;
  margin-bottom: 10px;
}

.search-box:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.training-runs-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
}

.training-runs-table th,
.training-runs-table td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

.training-runs-table th {
  background: #f8f9fa;
  font-weight: 600;
  color: #333;
  font-size: 14px;
}

.training-runs-table td {
  font-size: 14px;
  color: #666;
}

.training-run-name {
  color: #007bff;
  text-decoration: none;
  font-weight: 500;
}

.training-run-name:hover {
  text-decoration: underline;
}

.training-run-status {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
}

.training-run-status.running {
  background: #e3f2fd;
  color: #1976d2;
}

.training-run-status.completed {
  background: #e8f5e8;
  color: #2e7d32;
}

.training-run-status.failed {
  background: #ffebee;
  color: #c62828;
}

.training-run-user {
  font-family: monospace;
  font-size: 12px;
}

.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
  color: #666;
}

.error-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
  color: #c62828;
  text-align: center;
}
`

interface TrainingRunsProps {
  repo: Repo
}

export function TrainingRuns({ repo }: TrainingRunsProps) {
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    const loadTrainingRuns = async () => {
      try {
        setLoading(true)
        const response = await repo.getTrainingRuns()
        setTrainingRuns(response.training_runs)
        setError(null)
      } catch (err: any) {
        setError(`Failed to load training runs: ${err.message}`)
      } finally {
        setLoading(false)
      }
    }

    loadTrainingRuns()
  }, [repo])

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'running'
      case 'completed':
        return 'completed'
      case 'failed':
        return 'failed'
      default:
        return ''
    }
  }

  const filteredTrainingRuns = trainingRuns.filter(run => {
    const query = searchQuery.toLowerCase()
    return (
      run.name.toLowerCase().includes(query) ||
      run.status.toLowerCase().includes(query) ||
      run.user_id.toLowerCase().includes(query)
    )
  })

  if (loading) {
    return (
      <div className="training-runs-container">
        <style>{TRAINING_RUNS_CSS}</style>
        <div className="training-runs-content">
          <div className="loading-container">
            <div>Loading training runs...</div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="training-runs-container">
        <style>{TRAINING_RUNS_CSS}</style>
        <div className="training-runs-content">
          <div className="error-container">
            <div>{error}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="training-runs-container">
      <style>{TRAINING_RUNS_CSS}</style>
      <div className="training-runs-content">
        <div className="training-runs-header">
          <h1 className="training-runs-title">Training Runs</h1>
          <div>
            <input
              type="text"
              placeholder="Search training runs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-box"
            />
            <div className="training-runs-count">
              {searchQuery ?
                `${filteredTrainingRuns.length} of ${trainingRuns.length} training runs` :
                `${trainingRuns.length} training run${trainingRuns.length !== 1 ? 's' : ''}`
              }
            </div>
          </div>
        </div>

        {trainingRuns.length === 0 ? (
          <div className="loading-container">
            <div>No training runs found.</div>
          </div>
        ) : filteredTrainingRuns.length === 0 ? (
          <div className="loading-container">
            <div>No training runs match your search.</div>
          </div>
        ) : (
          <table className="training-runs-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Created</th>
                <th>Finished</th>
                <th>User</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrainingRuns.map((run) => (
                <tr key={run.id}>
                  <td>
                    <Link to={`/training-run/${run.id}`} className="training-run-name">
                      {run.name}
                    </Link>
                  </td>
                  <td>
                    <span className={`training-run-status ${getStatusClass(run.status)}`}>
                      {run.status}
                    </span>
                  </td>
                  <td>{formatDate(run.created_at)}</td>
                  <td>{run.finished_at ? formatDate(run.finished_at) : '—'}</td>
                  <td>
                    <span className="training-run-user">{run.user_id}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
