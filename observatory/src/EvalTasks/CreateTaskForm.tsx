import { FC, useState } from 'react'

import { Button } from '../components/Button'
import { Input } from '../components/Input'
import { Repo } from '../repo'

export const CreateTaskForm: FC<{
  repo: Repo
  setError: (error: string | null) => void
  onTaskCreated: () => void
}> = ({ repo, setError, onTaskCreated }) => {
  const [command, setCommand] = useState('')
  const [gitHash, setGitHash] = useState('')
  const [loading, setLoading] = useState(false)

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
      onTaskCreated()
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white p-6 rounded-md border border-gray-200 shadow-md mb-8">
      <h3 className="mt-0 mb-5">Create New Task</h3>

      <div className="flex gap-2 items-end">
        <div className="flex-1">
          <label className="block mb-1 text-sm font-medium">Command</label>
          <Input value={command} onChange={setCommand} placeholder="Enter command to execute" />
        </div>

        <div>
          <label className="block mb-1 text-sm font-medium">Git Hash (optional)</label>
          <Input value={gitHash} onChange={setGitHash} placeholder="Git commit hash" />
        </div>

        <Button onClick={handleCreateTask} disabled={loading || !command.trim()} theme="primary">
          {loading ? 'Creating...' : 'Create Task'}
        </Button>
      </div>
    </div>
  )
}
