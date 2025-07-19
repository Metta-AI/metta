import { useEffect, useState } from 'react'
import { Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom'
import { ServerRepo, Repo } from './repo'
import { Dashboard } from './Dashboard'
import { Episodes } from './Episodes'
import { TokenManager } from './TokenManager'
import { SavedDashboards } from './SavedDashboards'
import { SQLQuery } from './SQLQuery'
import { TrainingRuns } from './TrainingRuns'
import { TrainingRunDetail } from './TrainingRunDetail'
import { EvalTasks } from './EvalTasks'
import { config } from './config'

// CSS for navigation
const NAV_CSS = `
.nav-container {
  background: #fff;
  border-bottom: 1px solid #ddd;
  padding: 0 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.1);
}

.nav-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-brand {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.nav-tabs {
  display: flex;
  gap: 0;
}

.nav-tab {
  padding: 15px 20px;
  text-decoration: none;
  color: #666;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.nav-tab:hover {
  color: #333;
  background: #f8f9fa;
}

.nav-tab.active {
  color: #007bff;
  border-bottom-color: #007bff;
}

.page-container {
  padding-top: 0;
}
`

function App() {
  // Data loading state
  type DefaultState = {
    type: 'error'
    error: string | null
  }
  type LoadingState = {
    type: 'loading'
  }
  type RepoState = {
    type: 'repo'
    repo: Repo
    currentUser: string
  }
  type State = DefaultState | LoadingState | RepoState

  const [state, setState] = useState<State>({ type: 'loading' })
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const initializeRepo = async () => {
      const serverUrl = config.apiBaseUrl
      try {
        const repo = new ServerRepo(serverUrl)

        // Test the connection by calling getSuites
        await repo.getSuites()

        // Get current user
        const userInfo = await repo.whoami()
        const currentUser = userInfo.user_email

        setState({ type: 'repo', repo, currentUser })
      } catch (err: any) {
        setState({
          type: 'error',
          error: `Failed to connect to server: ${err.message}. Make sure the server is running at ${serverUrl}`,
        })
      }
    }

    initializeRepo()
  }, [navigate])

  const handleDashboardPageChange = () => {
    navigate('/dashboard')
  }

  if (state.type === 'error') {
    return (
      <div
        style={{
          fontFamily: 'Arial, sans-serif',
          margin: 0,
          padding: '20px',
          background: '#f8f9fa',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            maxWidth: '600px',
            margin: '0 auto',
            background: '#fff',
            padding: '40px',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,.1)',
            textAlign: 'center',
          }}
        >
          <h1
            style={{
              color: '#333',
              marginBottom: '20px',
            }}
          >
            Policy Evaluation Dashboard
          </h1>
          <p style={{ marginBottom: '20px', color: '#666' }}>Unable to connect to the evaluation server.</p>
          {state.error && <div style={{ color: 'red', marginTop: '10px', marginBottom: '20px' }}>{state.error}</div>}
          <p style={{ color: '#666', fontSize: '14px' }}>Please ensure the server is running and accessible.</p>
        </div>
      </div>
    )
  }

  if (state.type === 'loading') {
    return (
      <div
        style={{
          fontFamily: 'Arial, sans-serif',
          margin: 0,
          padding: '20px',
          background: '#f8f9fa',
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            textAlign: 'center',
            color: '#666',
          }}
        >
          <h2>Connecting to server...</h2>
          <p>Loading evaluation data from the server.</p>
        </div>
      </div>
    )
  }

  if (state.type === 'repo') {
    return (
      <div style={{ fontFamily: 'Arial, sans-serif', margin: 0 }}>
        <style>{NAV_CSS}</style>
        <nav className="nav-container">
          <div className="nav-content">
            <Link to="/dashboard" className="nav-brand" onClick={handleDashboardPageChange}>
              Policy Evaluation Dashboard
            </Link>
            <div className="nav-tabs">
              <Link
                to="/dashboard"
                className={`nav-tab ${location.pathname === '/dashboard' ? 'active' : ''}`}
                onClick={handleDashboardPageChange}
              >
                Dashboard
              </Link>
              <Link
                to="/training-runs"
                className={`nav-tab ${location.pathname.startsWith('/training-run') ? 'active' : ''}`}
              >
                Training Runs
              </Link>
              <Link to="/episodes" className={`nav-tab ${location.pathname === '/episodes' ? 'active' : ''}`}>
                Episodes
              </Link>
              <Link
                to="/eval-tasks"
                className={`nav-tab ${location.pathname.startsWith('/eval-task') ? 'active' : ''}`}
              >
                Evaluate Policies
              </Link>
              <Link to="/saved" className={`nav-tab ${location.pathname === '/saved' ? 'active' : ''}`}>
                Saved Dashboards
              </Link>
              <Link to="/tokens" className={`nav-tab ${location.pathname === '/tokens' ? 'active' : ''}`}>
                Token Management
              </Link>
              <Link to="/sql-query" className={`nav-tab ${location.pathname === '/sql-query' ? 'active' : ''}`}>
                SQL Query
              </Link>
            </div>
          </div>
        </nav>

        <div className="page-container">
          <Routes>
            <Route path="/dashboard" element={<Dashboard repo={state.repo} />} />
            <Route path="/training-runs" element={<TrainingRuns repo={state.repo} />} />
            <Route path="/training-run/:runId" element={<TrainingRunDetail repo={state.repo} />} />
            <Route path="/episodes" element={<Episodes repo={state.repo} />} />
            <Route path="/eval-tasks" element={<EvalTasks repo={state.repo} />} />
            <Route path="/saved" element={<SavedDashboards repo={state.repo} currentUser={state.currentUser} />} />
            <Route path="/tokens" element={<TokenManager repo={state.repo} />} />
            <Route path="/sql-query" element={<SQLQuery repo={state.repo} />} />
            <Route path="/" element={<Dashboard repo={state.repo} />} />
          </Routes>
        </div>
      </div>
    )
  }

  return null
}

export default App
