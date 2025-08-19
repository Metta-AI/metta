import { ActiveRun } from '../types'
import { useState, useEffect } from 'react'

interface ActiveRunsTableProps {
  activeRuns: ActiveRun[]
  isLoading?: boolean
}

export function ActiveRunsTable({ activeRuns, isLoading = false }: ActiveRunsTableProps) {
  const [lastUpdated, setLastUpdated] = useState(new Date())

  useEffect(() => {
    setLastUpdated(new Date())
  }, [activeRuns])

  const formatRuntime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`
    } else {
      return `${secs}s`
    }
  }

  const formatLastUpdate = (secondsSince: number | null | undefined) => {
    if (secondsSince === null || secondsSince === undefined) {
      return 'Unknown'
    }
    
    if (secondsSince < 60) {
      return `${secondsSince}s ago`
    } else if (secondsSince < 3600) {
      const minutes = Math.floor(secondsSince / 60)
      return `${minutes}m ago`
    } else {
      const hours = Math.floor(secondsSince / 3600)
      const minutes = Math.floor((secondsSince % 3600) / 60)
      return `${hours}h ${minutes}m ago`
    }
  }

  const getHeartbeatStatus = (secondsSince: number | null | undefined) => {
    if (secondsSince === null || secondsSince === undefined) {
      return { color: '#6c757d', status: 'unknown' }
    }
    
    if (secondsSince < 60) {
      return { color: '#28a745', status: 'healthy' }  // Green - healthy
    } else if (secondsSince < 300) {  // 5 minutes
      return { color: '#ffc107', status: 'delayed' }  // Yellow - delayed
    } else {
      return { color: '#dc3545', status: 'stale' }  // Red - possibly dead
    }
  }

  const formatTimesteps = (current: number, total: number) => {
    if (total >= 1000000) {
      return `${(current / 1000000).toFixed(2)}M / ${(total / 1000000).toFixed(1)}M`
    } else if (total >= 1000) {
      return `${(current / 1000).toFixed(1)}K / ${(total / 1000).toFixed(0)}K`
    }
    return `${current} / ${total}`
  }

  if (!activeRuns || activeRuns.length === 0) {
    return (
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>
            Active Training Runs
          </h3>
          <div style={{ fontSize: '12px', color: '#6c757d', display: 'flex', alignItems: 'center', gap: '8px' }}>
            {isLoading && <span className="spinner spinner-sm" />}
            <span>
              Last updated: {lastUpdated.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                hour12: true 
              })}
            </span>
          </div>
        </div>
        <div style={{ textAlign: 'center', color: '#6c757d', padding: '20px' }}>
          No active runs at the moment
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>
          Active Training Runs ({activeRuns.length})
        </h3>
        <div style={{ fontSize: '12px', color: '#6c757d', display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isLoading && <span className="spinner spinner-sm" />}
          <span>
            Last updated: {lastUpdated.toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit', 
              second: '2-digit',
              hour12: true 
            })}
          </span>
        </div>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="table">
          <thead>
            <tr>
              <th>Run Name</th>
              <th>Progress</th>
              <th>Timesteps</th>
              <th>Runtime</th>
              <th>Last Heartbeat</th>
              <th>Current Score</th>
              <th>Cost</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {activeRuns
              .sort((a, b) => b.timesteps - a.timesteps)
              .map(run => (
                <tr key={run.run_id}>
                  <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                    {run.run_name.length > 30 ? run.run_name.slice(0, 30) + '...' : run.run_name}
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        width: '100px',
                        height: '20px',
                        backgroundColor: '#e9ecef',
                        borderRadius: '10px',
                        overflow: 'hidden',
                        position: 'relative'
                      }}>
                        <div style={{
                          width: `${run.progress}%`,
                          height: '100%',
                          backgroundColor: run.progress > 75 ? '#28a745' : 
                                          run.progress > 50 ? '#ffc107' : 
                                          run.progress > 25 ? '#17a2b8' : '#007bff',
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                      <span style={{ fontSize: '12px', fontWeight: 500 }}>
                        {run.progress.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td>{formatTimesteps(run.timesteps, run.total_timesteps)}</td>
                  <td>{formatRuntime(run.runtime_seconds)}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: getHeartbeatStatus(run.seconds_since_update).color,
                        animation: run.seconds_since_update && run.seconds_since_update < 60 
                          ? 'pulse 2s infinite' 
                          : 'none'
                      }} />
                      <span style={{ 
                        fontSize: '12px',
                        color: getHeartbeatStatus(run.seconds_since_update).color
                      }}>
                        {formatLastUpdate(run.seconds_since_update)}
                      </span>
                    </div>
                  </td>
                  <td>{run.score.toFixed(4)}</td>
                  <td>${run.cost.toFixed(2)}</td>
                  <td>
                    <span className="badge badge-info">
                      {run.state}
                    </span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      
      {/* Auto-refresh indicator */}
      <div style={{ 
        marginTop: '12px', 
        fontSize: '12px', 
        color: '#6c757d',
        textAlign: 'right'
      }}>
        <span style={{ marginRight: '8px' }}>⟳</span>
        Auto-refreshes with page data
      </div>
    </div>
  )
}