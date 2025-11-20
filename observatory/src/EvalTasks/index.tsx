import { FC, useRef, useState } from 'react'
import { Repo } from '../repo'
import { TasksTable, TasksTableHandle } from './TasksTable'
import { CreateTaskForm } from './CreateTaskForm'

interface Props {
  repo: Repo
}

export const EvalTasks: FC<Props> = ({ repo }) => {
  const tasksTableRef = useRef<TasksTableHandle | null>(null)
  const [error, setError] = useState<string | null>(null)

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px' }}>Evaluation Tasks</h1>

      {error && (
        <div
          style={{
            padding: '10px 15px',
            marginBottom: '20px',
            backgroundColor: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
            color: '#721c24',
          }}
        >
          {error}
        </div>
      )}

      <CreateTaskForm repo={repo} setError={setError} onTaskCreated={() => tasksTableRef.current?.loadTasks(1)} />

      <TasksTable repo={repo} setError={setError} ref={tasksTableRef} />
    </div>
  )
}
