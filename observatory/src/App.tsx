import { Route, Routes } from 'react-router-dom'

import { AppProvider } from './AppContext'
import { EpisodeDetailPage } from './EpisodeDetailPage'
import { EvalTasks } from './EvalTasks/index'
import { Leaderboard } from './Leaderboard'
import { PoliciesPage } from './PoliciesPage'
import { PolicyPage } from './PolicyPage'
import { PolicyVersionPage } from './PolicyVersionPage'
import { SQLQuery } from './SQLQuery'
import { TopMenu } from './TopMenu'

function App() {
  return (
    <AppProvider>
      <div>
        <TopMenu />

        <div>
          <Routes>
            <Route path="/" element={<PoliciesPage />} />
            <Route path="/policies" element={<PoliciesPage />} />
            <Route path="/policies/:policyId" element={<PolicyPage />} />
            <Route path="/policies/versions/:policyVersionId" element={<PolicyVersionPage />} />
            <Route path="/eval-tasks" element={<EvalTasks />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/episodes/:episodeId" element={<EpisodeDetailPage />} />
            <Route path="/sql-query" element={<SQLQuery />} />
          </Routes>
        </div>
      </div>
    </AppProvider>
  )
}

export default App
