import clsx from 'clsx'
import { useEffect, useState } from 'react'
import { Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom'

import { config } from './config'
import { EvalTasks } from './EvalTasks/index'
import { Leaderboard } from './Leaderboard'
import { Repo } from './repo'
import { SQLQuery } from './SQLQuery'

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
        const repo = new Repo(serverUrl)

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

  switch (state.type) {
    case 'error':
      return (
        <div
          style={{
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

    case 'loading':
      return (
        <div
          style={{
            margin: 0,
            padding: '20px',
            background: '#f8f9fa',
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div className="text-center text-gray-600">
            <h2>Connecting to server...</h2>
            <p>Loading evaluation data from the server.</p>
          </div>
        </div>
      )

    case 'repo':
      return (
        <div style={{ fontFamily: 'Arial, sans-serif' }}>
          <style>{NAV_CSS}</style>
          <nav className="nav-container">
            <div className="nav-content">
              <div className="nav-tabs">
                <Link
                  to="/eval-tasks"
                  className={clsx('nav-tab', location.pathname.startsWith('/eval-task') && 'active')}
                >
                  Evaluate Policies
                </Link>
                <Link to="/leaderboard" className={clsx('nav-tab', location.pathname === '/leaderboard' && 'active')}>
                  Leaderboard
                </Link>
                <Link to="/sql-query" className={clsx('nav-tab', location.pathname === '/sql-query' && 'active')}>
                  SQL Query
                </Link>
              </div>
            </div>
          </nav>

          <div>
            <Routes>
              <Route path="/eval-tasks" element={<EvalTasks repo={state.repo} />} />
              <Route path="/leaderboard" element={<Leaderboard repo={state.repo} currentUser={state.currentUser} />} />
              <Route path="/sql-query" element={<SQLQuery repo={state.repo} />} />
              <Route path="/" element={<EvalTasks repo={state.repo} />} />
            </Routes>
          </div>
        </div>
      )
    default:
      throw new Error(`Unknown state type: ${state satisfies never}`)
  }
}

export default App
