import React, { useState } from 'react'
import { EvalTask } from '../repo'
import TypeaheadInput from './TypeaheadInput'
import { getRecentPolicies, getRecentSimSuites } from '../utils/evalTasks'

interface CreateTaskFormProps {
  repo: any
  tasks: EvalTask[]
  onTaskCreated: () => Promise<void>
}

export const CreateTaskForm: React.FC<CreateTaskFormProps> = ({ repo, tasks, onTaskCreated }) => {
  const [policyIdInput, setPolicyIdInput] = useState<string>('')
  const [gitHash, setGitHash] = useState<string>('')
  const [simSuite, setSimSuite] = useState<string>('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleCreateTask = async () => {
    if (!policyIdInput.trim()) {
      setError('Please enter a policy name or ID')
      return
    }

    setLoading(true)
    setError(null)

    try {
      let policyId = policyIdInput.trim()

      // Check if input looks like a UUID
      const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(policyId)

      if (!isUuid) {
        // Try to resolve policy name to ID
        try {
          const policyIds = await repo.getPolicyIds([policyId])
          if (policyIds[policyId]) {
            policyId = policyIds[policyId]
          } else {
            setError(`Policy with name '${policyId}' not found`)
            return
          }
        } catch (e) {
          setError(`Failed to resolve policy name: ${e}`)
          return
        }
      }

      await repo.createEvalTask({
        policy_id: policyId,
        git_hash: gitHash || null,
        sim_suite: simSuite,
        env_overrides: {},
      })

      // Clear form
      setPolicyIdInput('')
      setGitHash('')

      // Refresh tasks
      await onTaskCreated()
    } catch (err: any) {
      setError(`Failed to create task: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
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
      <h3
        style={{
          marginTop: 0,
          marginBottom: '20px',
          fontSize: '18px',
          fontWeight: 600,
          color: '#1a1a1a',
        }}
      >
        Create New Evaluation Task
      </h3>

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

      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
        <div style={{ flex: '1 1 300px' }}>
          <label
            style={{
              display: 'block',
              marginBottom: '6px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#555',
            }}
          >
            Policy Name or ID
          </label>
          <TypeaheadInput
            value={policyIdInput}
            onChange={setPolicyIdInput}
            placeholder="Enter policy name or ID"
            suggestions={getRecentPolicies(tasks)}
            maxSuggestions={10}
            filterType="substring"
          />
        </div>

        <div style={{ flex: '1 1 250px' }}>
          <label
            style={{
              display: 'block',
              marginBottom: '6px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#555',
            }}
          >
            Git Commit
          </label>
          <input
            type="text"
            value={gitHash}
            onChange={(e) => setGitHash(e.target.value)}
            placeholder="Latest main (default)"
            style={{
              width: '100%',
              padding: '10px 12px',
              borderRadius: '6px',
              border: '1px solid #d1d5db',
              fontSize: '14px',
              backgroundColor: '#fff',
              transition: 'border-color 0.2s',
              outline: 'none',
            }}
            onFocus={(e) => (e.target.style.borderColor = '#007bff')}
            onBlur={(e) => (e.target.style.borderColor = '#d1d5db')}
          />
        </div>

        <div style={{ flex: '0 0 140px' }}>
          <label
            style={{
              display: 'block',
              marginBottom: '6px',
              fontSize: '13px',
              fontWeight: 500,
              color: '#555',
            }}
          >
            Suite
          </label>
          <TypeaheadInput
            value={simSuite}
            onChange={setSimSuite}
            placeholder="Enter sim suite"
            suggestions={getRecentSimSuites(tasks)}
            maxSuggestions={5}
            filterType="prefix"
          />
        </div>

        <button
          onClick={handleCreateTask}
          disabled={loading || !policyIdInput.trim()}
          style={{
            padding: '10px 24px',
            backgroundColor: loading || !policyIdInput.trim() ? '#9ca3af' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: loading || !policyIdInput.trim() ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 500,
            transition: 'background-color 0.2s',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={(e) => {
            if (!loading && policyIdInput.trim()) {
              e.currentTarget.style.backgroundColor = '#0056b3'
            }
          }}
          onMouseLeave={(e) => {
            if (!loading && policyIdInput.trim()) {
              e.currentTarget.style.backgroundColor = '#007bff'
            }
          }}
        >
          {loading ? 'Creating...' : 'Create Task'}
        </button>
      </div>
    </div>
  )
}
