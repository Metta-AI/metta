import { FC, useRef, useState } from 'react'
import { Repo } from '../repo'
import { TasksTable, TasksTableHandle } from './TasksTable'

interface Props {
  repo: Repo
}

export const EvalTasks: FC<Props> = ({ repo }) => {
  // Form state
  const [command, setCommand] = useState('')
  const [gitHash, setGitHash] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const tasksTableRef = useRef<TasksTableHandle | null>(null)

  // Create task
  const handleCreateTask = async () => {
    if (!command.trim()) {
      setError('Please enter a command')
      return
    }

    setLoading(true)
    setError(null)

    try {
      await repo.createEvalTask({
        command: command.trim(),
        git_hash: gitHash.trim() || null,
        attributes: {},
      })

      setCommand('')
      setGitHash('')
      tasksTableRef.current?.loadTasks(1)
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

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

      {/* Create Task Section */}
      <div
        style={{
          backgroundColor: '#ffffff',
          padding: '24px',
          borderRadius: '12px',
          marginBottom: '30px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          border: '1px solid #e8e8e8',
        }}
      >
        <h3 className="mt-0 mb-5">Create New Task</h3>

        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="block mb-1 text-sm font-medium">Command</label>
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="Enter command to execute"
              className="box-border"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
              }}
            />
          </div>

          <div>
            <label className="block mb-1 text-sm font-medium">Git Hash (optional)</label>
            <input
              type="text"
              value={gitHash}
              onChange={(e) => setGitHash(e.target.value)}
              placeholder="Git commit hash"
              className="box-border"
              style={{
                width: '100%',
                padding: '10px 12px',
                borderRadius: '6px',
                border: '1px solid #d1d5db',
                fontSize: '14px',
              }}
            />
          </div>

          <button
            onClick={handleCreateTask}
            disabled={loading || !command.trim()}
            style={{
              padding: '10px 24px',
              backgroundColor: loading || !command.trim() ? '#9ca3af' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !command.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 500,
            }}
          >
            {loading ? 'Creating...' : 'Create Task'}
          </button>
        </div>
      </div>

      <TasksTable repo={repo} setError={setError} ref={tasksTableRef} />
    </div>
  )
}
