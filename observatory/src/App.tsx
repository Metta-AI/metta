import { Route, Routes } from 'react-router-dom'

import { AppProvider } from './AppContext'
import { EvalTasks } from './EvalTasks/index'
import { Leaderboard } from './Leaderboard'
import { SQLQuery } from './SQLQuery'
import { TopMenu } from './TopMenu'

function App() {
  return (
    <AppProvider>
      <div>
        <TopMenu />

        <div>
          <Routes>
            <Route path="/" element={<EvalTasks />} />
            <Route path="/eval-tasks" element={<EvalTasks />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/sql-query" element={<SQLQuery />} />
          </Routes>
        </div>
      </div>
    </AppProvider>
  )
}

export default App
