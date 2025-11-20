import { FC, useRef, useState } from 'react'

import { Repo } from '../repo'
import { CreateTaskForm } from './CreateTaskForm'
import { TasksTable, TasksTableHandle } from './TasksTable'

interface Props {
  repo: Repo
}

export const EvalTasks: FC<Props> = ({ repo }) => {
  const tasksTableRef = useRef<TasksTableHandle | null>(null)
  const [error, setError] = useState<string | null>(null)

  return (
    <div className="p-5 max-w-[1400px] mx-auto">
      <h1 className="mb-5">Evaluation Tasks</h1>

      {error && (
        <div className="px-4 py-3 mb-5 text-sm bg-red-50 border border-red-400 text-red-800 rounded">{error}</div>
      )}

      <CreateTaskForm repo={repo} setError={setError} onTaskCreated={() => tasksTableRef.current?.loadTasks(1)} />

      <TasksTable repo={repo} setError={setError} ref={tasksTableRef} />
    </div>
  )
}
