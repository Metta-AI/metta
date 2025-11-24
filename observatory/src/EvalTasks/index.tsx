import { FC, useContext, useRef, useState } from 'react'

import { AppContext } from '../AppContext'
import { CreateTaskForm } from './CreateTaskForm'
import { TasksTable, TasksTableHandle } from './TasksTable'

export const EvalTasks: FC = () => {
  const { repo } = useContext(AppContext)
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
