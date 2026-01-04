import { useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'

import { AppProvider } from './AppContext'
import { hasToken, initiateLogin } from './auth'
import { AuthCallback } from './AuthCallback'
import { EpisodeDetailPage } from './EpisodeDetailPage'
import { EpisodeJobs } from './EpisodeJobs'
import { EvalTasks } from './EvalTasks/index'
import { Leaderboard } from './Leaderboard'
import { PoliciesPage } from './PoliciesPage'
import { PolicyPage } from './PolicyPage'
import { PolicyVersionPage } from './PolicyVersionPage'
import { PlayerPage } from './Seasons/PlayerPage'
import { SeasonsPage } from './Seasons/index'
import { SQLQuery } from './SQLQuery'
import { TopMenu } from './TopMenu'

function App() {
  useEffect(() => {
    // Check if we have a token on app initialization
    if (!hasToken()) {
      // Don't redirect if we're on the callback page
      if (!window.location.pathname.startsWith('/auth/callback')) {
        initiateLogin()
      }
    }
  }, [])

  return (
    <AppProvider>
      <div className="min-h-screen font-sans flex flex-col">
        <TopMenu />

        <div className="bg-gray-50 flex-1">
          <Routes>
            <Route path="/auth/callback" element={<AuthCallback />} />
            <Route path="/" element={<PoliciesPage />} />
            <Route path="/policies" element={<PoliciesPage />} />
            <Route path="/policies/:policyId" element={<PolicyPage />} />
            <Route path="/policies/versions/:policyVersionId" element={<PolicyVersionPage />} />
            <Route path="/eval-tasks" element={<EvalTasks />} />
            <Route path="/episode-jobs" element={<EpisodeJobs />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/seasons" element={<SeasonsPage />} />
            <Route path="/seasons/:seasonName" element={<SeasonsPage />} />
            <Route path="/seasons/:seasonName/players/:policyVersionId" element={<PlayerPage />} />
            <Route path="/episodes/:episodeId" element={<EpisodeDetailPage />} />
            <Route path="/sql-query" element={<SQLQuery />} />
          </Routes>
        </div>
      </div>
    </AppProvider>
  )
}

export default App
