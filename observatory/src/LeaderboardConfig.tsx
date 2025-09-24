import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Repo, LeaderboardCreateOrUpdate, UnifiedPolicyInfo } from './repo'
import { EvalSelector } from './components/EvalSelector'
import { MetricSelector } from './components/MetricSelector'
import { DateSelector } from './components/DateSelector'

// CSS for leaderboard config
const LEADERBOARD_CONFIG_CSS = `
.config-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
}

.config-header {
  text-align: center;
  margin-bottom: 30px;
}

.config-title {
  color: #333;
  margin-bottom: 10px;
}

.config-subtitle {
  color: #666;
  font-size: 14px;
}

.form-group {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: #333;
}

.form-input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  width: 100%;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0,123,255,.25);
}

.form-textarea {
  min-height: 60px;
  resize: vertical;
}

.form-checkbox-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.form-checkbox {
  width: auto;
}

.form-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn-primary {
  background: #007bff;
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  background: #0056b3;
}

.btn-secondary {
  background: #6c757d;
  color: #fff;
}

.btn-secondary:hover:not(:disabled) {
  background: #5a6268;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.widget {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}

.widget-title {
  color: #333;
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.error {
  text-align: center;
  padding: 40px;
  color: #dc3545;
}

.small-input {
  max-width: 120px;
}
`

interface LeaderboardConfigProps {
  repo: Repo
}

export function LeaderboardConfig({ repo }: LeaderboardConfigProps) {
  const navigate = useNavigate()
  const { leaderboardId } = useParams<{ leaderboardId: string }>()
  const isEditMode = Boolean(leaderboardId)

  const formatDate = (date: Date) => {
    return date.toISOString().split('T')[0]
  }

  // Form state
  const [name, setName] = useState('')
  const [startDate, setStartDate] = useState(() => {
    const date = new Date() // default to today
    return formatDate(date)
  })
  const [selectedEvalNames, setSelectedEvalNames] = useState<Set<string>>(new Set())
  const [selectedMetric, setSelectedMetric] = useState<string>('reward')

  // Data state
  const [policies, setPolicies] = useState<UnifiedPolicyInfo[]>([])
  const [evalNames, setEvalNames] = useState<Set<string>>(new Set())
  const [availableMetrics, setAvailableMetrics] = useState<string[]>([])

  // UI state
  const [loading, setLoading] = useState({
    page: false,
    policies: false,
    evalCategories: false,
    metrics: false,
    saving: false,
  })
  const [error, setError] = useState<string | null>(null)

  const startDateDate = new Date(startDate)
  // Get filtered policies based on start date
  const filteredPolicies = policies.filter((policy) => {
    const policyDate = new Date(policy.created_at)
    return policyDate >= startDateDate
  })

  const trainingRunIds = filteredPolicies.filter((p) => p.type === 'training_run').map((p) => p.id)

  const runFreePolicyIds = filteredPolicies.filter((p) => p.type === 'policy').map((p) => p.id)

  // Load leaderboard data if in edit mode
  useEffect(() => {
    if (isEditMode && leaderboardId) {
      loadLeaderboard()
    }
  }, [isEditMode, leaderboardId])

  // Load policies on mount
  useEffect(() => {
    loadPolicies()
  }, [])

  // Load eval names when policies/start date changes
  useEffect(() => {
    if (filteredPolicies.length > 0) {
      loadEvalNames()
    } else {
      setEvalNames(new Set())
      setSelectedEvalNames(new Set())
    }
  }, [startDate, policies])

  // Load metrics when policies and evals are selected
  useEffect(() => {
    if (filteredPolicies.length > 0 && selectedEvalNames.size > 0) {
      loadMetrics()
    } else {
      setAvailableMetrics([])
      setSelectedMetric('reward')
    }
  }, [startDate, policies, selectedEvalNames])

  const loadLeaderboard = async () => {
    if (!leaderboardId) return

    try {
      setLoading((prev) => ({ ...prev, page: true }))
      setError(null)
      const data = await repo.getLeaderboard(leaderboardId)

      // Populate form with leaderboard data
      setName(data.name)
      setStartDate(data.start_date)
      setSelectedEvalNames(new Set(data.evals))
      setSelectedMetric(data.metric)
    } catch (err: any) {
      setError(`Failed to load leaderboard: ${err.message}`)
    } finally {
      setLoading((prev) => ({ ...prev, page: false }))
    }
  }

  const loadPolicies = async () => {
    try {
      setLoading((prev) => ({ ...prev, policies: true }))
      setError(null)
      const response = await repo.getPolicies()
      setPolicies(response.policies)
    } catch (err: any) {
      setError(`Failed to load policies: ${err.message}`)
    } finally {
      setLoading((prev) => ({ ...prev, policies: false }))
    }
  }

  const loadEvalNames = async () => {
    try {
      setLoading((prev) => ({ ...prev, evalCategories: true }))
      setError(null)
      const evalNamesData = await repo.getEvalNames({
        training_run_ids: trainingRunIds,
        run_free_policy_ids: runFreePolicyIds,
      })
      setEvalNames(evalNamesData)
      setSelectedEvalNames((prev) => new Set([...prev].filter((evalName) => evalNamesData.has(evalName))))
    } catch (err: any) {
      setError(`Failed to load eval names: ${err.message}`)
      setEvalNames(new Set())
      setSelectedEvalNames(new Set())
    } finally {
      setLoading((prev) => ({ ...prev, evalCategories: false }))
    }
  }

  const loadMetrics = async () => {
    try {
      setLoading((prev) => ({ ...prev, metrics: true }))
      setError(null)
      const metricsData = await repo.getAvailableMetrics({
        training_run_ids: trainingRunIds,
        run_free_policy_ids: runFreePolicyIds,
        eval_names: Array.from(selectedEvalNames),
      })
      setAvailableMetrics(metricsData)

      if (selectedMetric && !metricsData.includes(selectedMetric)) {
        if (metricsData.includes('reward')) {
          setSelectedMetric('reward')
        } else if (metricsData.length > 0) {
          setSelectedMetric(metricsData[0])
        } else {
          setSelectedMetric('')
        }
      }
    } catch (err: any) {
      setError(`Failed to load metrics: ${err.message}`)
      setAvailableMetrics([])
      setSelectedMetric('')
    } finally {
      setLoading((prev) => ({ ...prev, metrics: false }))
    }
  }

  const handleSave = async () => {
    if (!isEditMode && !name.trim()) {
      alert('Please enter a name for the leaderboard')
      return
    }

    if (selectedEvalNames.size === 0) {
      alert('Please select at least one evaluation')
      return
    }

    if (!selectedMetric) {
      alert('Please select a metric')
      return
    }

    try {
      setLoading((prev) => ({ ...prev, saving: true }))
      const data: LeaderboardCreateOrUpdate = {
        name: name.trim(),
        evals: Array.from(selectedEvalNames),
        metric: selectedMetric,
        start_date: startDate,
      }

      if (isEditMode && leaderboardId) {
        await repo.updateLeaderboard(leaderboardId, data)
      } else {
        // Create new leaderboard
        await repo.createLeaderboard(data)
      }

      navigate('/leaderboards')
    } catch (err: any) {
      alert(`Failed to ${isEditMode ? 'update' : 'create'} leaderboard: ${err.message}`)
    } finally {
      setLoading((prev) => ({ ...prev, saving: false }))
    }
  }

  const handleCancel = () => {
    navigate('/leaderboards')
  }

  if (loading.page) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_CONFIG_CSS}</style>
        <div className="loading">
          <h3>Loading leaderboard...</h3>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
        <style>{LEADERBOARD_CONFIG_CSS}</style>
        <div className="error">
          <h3>Error</h3>
          <p>{error}</p>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px', background: '#f8f9fa', minHeight: 'calc(100vh - 60px)' }}>
      <style>{LEADERBOARD_CONFIG_CSS}</style>

      <div className="config-container">
        <div className="config-header">
          <h1 className="config-title">{isEditMode ? 'Edit Leaderboard' : 'Create New Leaderboard'}</h1>
          <p className="config-subtitle">
            {isEditMode
              ? 'Update leaderboard settings to adjust policy performance tracking.'
              : 'Configure leaderboard settings to track and compare policy performance across evaluations.'}
          </p>
        </div>

        {/* Basic Information */}
        <div className="form-group">
          <label className="form-label" htmlFor="name">
            Leaderboard Name
          </label>
          <input
            id="name"
            type="text"
            className="form-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter a descriptive name for your leaderboard"
          />
        </div>

        {/* Start Date Filter */}
        <div className="widget">
          <h3 className="widget-title">Policy Filter</h3>
          <DateSelector value={startDate} onChange={setStartDate} label="Training Start Date" />
          <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>
            Only policies created after this date will be included. Currently showing {filteredPolicies.length}{' '}
            policies.
          </p>
        </div>

        {/* Evaluation Selection */}
        <div className="widget">
          <h3 className="widget-title">Evaluation Selection</h3>
          <EvalSelector
            evalNames={evalNames}
            selectedEvalNames={selectedEvalNames}
            onSelectionChange={setSelectedEvalNames}
            loading={loading.evalCategories}
          />
        </div>

        {/* Training Run Policy Selector - Removed from leaderboard creation */}

        {/* Metric Selection */}
        <div className="widget">
          <h3 className="widget-title">Metric Selection</h3>
          <MetricSelector
            metrics={availableMetrics}
            selectedMetric={selectedMetric}
            onSelectionChange={setSelectedMetric}
            loading={loading.metrics}
            disabled={filteredPolicies.length === 0 || selectedEvalNames.size === 0}
          />
        </div>

        {/* Advanced Options - Removed from leaderboard creation */}

        {/* Form Actions */}
        <div className="form-actions">
          <button className="btn btn-secondary" onClick={handleCancel} disabled={loading.saving}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={
              loading.saving || (!isEditMode && !name.trim()) || selectedEvalNames.size === 0 || !selectedMetric
            }
          >
            {loading.saving
              ? isEditMode
                ? 'Updating...'
                : 'Creating...'
              : isEditMode
                ? 'Update Leaderboard'
                : 'Create Leaderboard'}
          </button>
        </div>
      </div>
    </div>
  )
}
