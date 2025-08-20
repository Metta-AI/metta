import React, { useState, useEffect, useCallback } from 'react'
import { SkyJob } from '../types'
import { fetchSkyJobs, cancelSkyJob, launchWorker } from '../api/skyApi'

interface SkyJobsMonitorProps {
  sweepName: string
}

export function SkyJobsMonitor({ sweepName }: SkyJobsMonitorProps) {
  const [jobs, setJobs] = useState<SkyJob[]>([])
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set())
  const [showGpuModal, setShowGpuModal] = useState(false)
  const [selectedGpus, setSelectedGpus] = useState('4')
  const [launchingWorker, setLaunchingWorker] = useState(false)

  const refreshJobs = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchSkyJobs()
      setJobs(data)
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshJobs()
  }, [refreshJobs])

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(refreshJobs, 30000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshJobs])

  const handleCancelJob = async (jobId: string) => {
    if (confirm(`Are you sure you want to cancel job ${jobId}?`)) {
      try {
        await cancelSkyJob(jobId)
        setTimeout(refreshJobs, 3000)
      } catch (error) {
        console.error('Failed to cancel job:', error)
      }
    }
  }

  const handleLaunchWorker = async () => {
    setLaunchingWorker(true)
    try {
      await launchWorker(parseInt(selectedGpus), sweepName)
      setShowGpuModal(false)
      setTimeout(refreshJobs, 5000)
    } catch (error) {
      console.error('Failed to launch worker:', error)
    } finally {
      setLaunchingWorker(false)
    }
  }

  const toggleJobLogs = (jobId: string) => {
    const newExpanded = new Set(expandedJobs)
    if (newExpanded.has(jobId)) {
      newExpanded.delete(jobId)
    } else {
      newExpanded.add(jobId)
    }
    setExpandedJobs(newExpanded)
  }

  const runningJobs = jobs.filter(j => j.status === 'RUNNING').length

  return (
    <div className="card">
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>
          Sky Jobs Monitor
        </h3>
        
        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <button 
            className="btn btn-primary btn-sm"
            onClick={refreshJobs}
            disabled={loading}
          >
            {loading && <div className="spinner spinner-sm" />}
            Refresh Jobs Status
          </button>
          
          <button 
            className="btn btn-success btn-sm"
            onClick={() => setShowGpuModal(true)}
          >
            Start New Worker
          </button>
          
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (30s)
          </label>
        </div>

        <div className="alert alert-info">
          <strong>Sky Jobs Status</strong>
          <div>In progress tasks: {runningJobs}</div>
        </div>

        {loading && (
          <div className="alert alert-primary">
            <div className="spinner spinner-sm" />
            Refreshing jobs status...
          </div>
        )}

        {jobs.length === 0 && !loading ? (
          <div style={{ textAlign: 'center', color: '#6c757d', padding: '20px' }}>
            No running jobs
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Resources</th>
                <th>Submitted</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <React.Fragment key={job.id}>
                  <tr>
                    <td style={{ fontFamily: 'monospace' }}>{job.id}</td>
                    <td>{job.name}</td>
                    <td>{job.resources}</td>
                    <td>{job.submitted}</td>
                    <td>{job.totalDuration}</td>
                    <td>
                      <span className={`badge badge-${
                        job.status === 'RUNNING' ? 'success' : 
                        job.status === 'SUCCEEDED' ? 'info' : 
                        'warning'
                      }`}>
                        {job.status}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => toggleJobLogs(job.id)}
                        style={{ marginRight: '8px' }}
                      >
                        {expandedJobs.has(job.id) ? 'Hide' : 'Show'} logs
                      </button>
                      {job.status === 'RUNNING' && (
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleCancelJob(job.id)}
                        >
                          Cancel
                        </button>
                      )}
                    </td>
                  </tr>
                  {expandedJobs.has(job.id) && (
                    <tr>
                      <td colSpan={7} style={{ backgroundColor: '#f8f9fa', padding: '16px' }}>
                        <pre style={{
                          backgroundColor: '#1e1e1e',
                          color: '#d4d4d4',
                          padding: '12px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          maxHeight: '200px',
                          overflow: 'auto'
                        }}>
                          Logs would appear here...
                        </pre>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {showGpuModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '400px',
            width: '100%'
          }}>
            <h4 style={{ marginBottom: '16px' }}>Launch Worker Configuration</h4>
            <p style={{ marginBottom: '16px', color: '#6c757d' }}>
              Choose the number of GPUs for the parallel worker:
            </p>
            <select
              className="form-control"
              value={selectedGpus}
              onChange={(e) => setSelectedGpus(e.target.value)}
              style={{ marginBottom: '20px' }}
            >
              {[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32].map(n => (
                <option key={n} value={n}>{n} GPUs</option>
              ))}
            </select>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setShowGpuModal(false)}
                disabled={launchingWorker}
              >
                Cancel
              </button>
              <button
                className="btn btn-success"
                onClick={handleLaunchWorker}
                disabled={launchingWorker}
              >
                {launchingWorker && <div className="spinner spinner-sm" />}
                Launch Worker
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}