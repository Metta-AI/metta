import { FC, useContext, useState } from 'react'

import { AppContext } from '../AppContext'
import { Card } from '../components/Card'
import { TasksTable } from './TasksTable'

export const EvalTasks: FC = () => {
  const { repo } = useContext(AppContext)
  const [error, setError] = useState<string | null>(null)

  return (
    <div className="p-5 max-w-[1400px] mx-auto">
      {error && (
        <div className="px-4 py-3 mb-5 text-sm bg-red-50 border border-red-400 text-red-800 rounded">{error}</div>
      )}

      <Card title="Remote Jobs">
        <TasksTable repo={repo} setError={setError} />
      </Card>
    </div>
  )
}
