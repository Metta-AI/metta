import { useState, useEffect, useCallback } from 'react';
import type { Experiment, Checkpoint, Job, HealthData } from '../types';
import { useApi } from '../hooks/useApi';
import { useNotifications } from '../hooks/useNotifications';
import { ScrollPanel } from './ScrollPanel';

interface ExpandedDetailsProps {
  experiment: Experiment;
  onSetEditingConfig: (id: string, editing: boolean) => void;
  onRefreshData: () => void;
  health?: HealthData | null;
}

function abbreviateStatus(status: string): string {
  const statusMap: Record<string, string> = {
    running: 'R',
    stopped: 'S',
    pending: 'P',
    starting: 'P',   // Pending (starting up)
    failed: 'F',
    succeeded: 'D',  // Done
    init: 'S',       // Stopped
    terminated: 'T',
    cancelled: 'S',  // Stopped
    unknown: '?',
  };
  const lower = status.toLowerCase();
  return statusMap[lower] || status.charAt(0).toUpperCase();
}

function formatDuration(startedAt: string | null, endedAt: string | null): string {
  if (!startedAt) return '-';
  const start = new Date(startedAt);
  // Use ended_at if available (completed jobs), otherwise use now (running jobs)
  const end = endedAt ? new Date(endedAt) : new Date();
  const diffMs = end.getTime() - start.getTime();

  if (diffMs < 0) return '-';

  const hours = Math.floor(diffMs / 3600000);
  const mins = Math.floor((diffMs % 3600000) / 60000);

  if (hours > 0) return `${hours}h ${mins}m`;
  if (mins > 0) return `${mins}m`;
  return '<1m';
}

function formatAge(timestamp: string | null): string {
  if (!timestamp) return '-';
  const start = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();

  if (diffMs < 60000) return 'just now';

  const mins = Math.floor(diffMs / 60000);
  if (mins < 60) return `${mins}m ago`;

  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatStaleness(seconds: number | null): string | null {
  if (seconds === null || seconds <= 30) return null;

  const mins = Math.floor(seconds / 60);
  if (mins < 60) return `${mins}m`;

  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h`;

  const days = Math.floor(hours / 24);
  return `${days}d`;
}

function buildCommand(exp: Experiment): string {
  const parts = [exp.base_command];
  parts.push(`--gpus=${exp.gpus}`);
  parts.push(`--nodes=${exp.nodes}`);
  if (exp.git_branch && exp.git_branch !== '-') {
    parts.push(`--git-ref=${exp.git_branch}`);
  }
  if (exp.tool_path) parts.push(exp.tool_path);
  parts.push(`run=${exp.name}`);

  const sortedFlags = Object.entries(exp.flags || {}).sort((a, b) => a[0].localeCompare(b[0]));
  for (const [key, value] of sortedFlags) {
    if (typeof value === 'boolean') {
      parts.push(`${key}=${value.toString().toLowerCase()}`);
    } else if (typeof value === 'string' && value.includes(' ')) {
      parts.push(`${key}="${value}"`);
    } else {
      parts.push(`${key}=${value}`);
    }
  }
  return parts.join(' ');
}

export function ExpandedDetails({ experiment, onSetEditingConfig, onRefreshData, health }: ExpandedDetailsProps) {
  const { apiCall } = useApi();
  const { showNotification } = useNotifications();

  // Copy to clipboard with toast notification
  // If url is provided and Cmd/Ctrl+click, open in new tab instead
  const handleCopy = useCallback((text: string, label: string, url?: string) => {
    return (e: React.MouseEvent) => {
      e.stopPropagation();
      if (url && (e.metaKey || e.ctrlKey)) {
        window.open(url, '_blank');
      } else {
        navigator.clipboard.writeText(text);
        showNotification(`Copied ${label}`, 'success');
      }
    };
  }, [showNotification]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [editedConfig, setEditedConfig] = useState<Record<string, string | number | null>>({});
  const [editedFlags, setEditedFlags] = useState<Record<string, unknown>>({});
  const [flagDefinitions, setFlagDefinitions] = useState<Array<{ flag: string; type: string; default: unknown; required: boolean }>>([]);
  const [newFlagSearch, setNewFlagSearch] = useState('');
  const [showFlagDropdown, setShowFlagDropdown] = useState(false);

  // Load jobs and checkpoints
  useEffect(() => {
    const loadData = async () => {
      try {
        const [jobsData, checkpointsData] = await Promise.all([
          apiCall<{ jobs: Job[] }>(`/experiments/${experiment.id}/jobs?limit=10`),
          apiCall<{ checkpoints: Checkpoint[] }>(`/experiments/${experiment.id}/checkpoints?limit=20`),
        ]);
        setJobs(jobsData.jobs || []);
        setCheckpoints(checkpointsData.checkpoints || []);
      } catch (error) {
        console.error('Error loading experiment details:', error);
      }
    };
    loadData();
  }, [apiCall, experiment.id]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setShowFlagDropdown(false);
    };
    if (showFlagDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showFlagDropdown]);

  const startEditing = async () => {
    setIsEditing(true);
    setEditedConfig({
      name: experiment.name,
      tool_path: experiment.tool_path,
      nodes: experiment.nodes,
      gpus: experiment.gpus,
      git_branch: experiment.git_branch,
    });
    setEditedFlags({ ...experiment.flags });
    onSetEditingConfig(experiment.id, true);

    // Load flag definitions for typeahead
    try {
      const toolPath = experiment.tool_path || '';
      const params = toolPath ? `?tool_path=${encodeURIComponent(toolPath)}` : '';
      console.log('[ExpandedDetails] Loading flags with params:', params);
      const response = await apiCall<{ flags: Array<{ flag: string; type: string; default: unknown; required: boolean }> }>(`/flags${params}`);
      console.log('[ExpandedDetails] Loaded flag definitions:', response.flags?.length);
      setFlagDefinitions(response.flags || []);
    } catch (error) {
      console.error('[ExpandedDetails] Error loading flag definitions:', error);
      setFlagDefinitions([]);
    }
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setNewFlagSearch('');
    setShowFlagDropdown(false);
    onSetEditingConfig(experiment.id, false);
  };

  const saveConfig = async () => {
    try {
      await apiCall(`/experiments/${experiment.id}`, {
        method: 'PATCH',
        body: JSON.stringify(editedConfig),
      });
      // Filter out reserved keys that should be top-level experiment fields, not flags
      const reservedKeys = ['gpus', 'nodes', 'name', 'tool_path', 'git_branch', 'instance_type', 'cloud', 'spot'];
      const cleanedFlags = Object.fromEntries(
        Object.entries(editedFlags).filter(([key]) => !reservedKeys.includes(key))
      );
      await apiCall(`/experiments/${experiment.id}/flags`, {
        method: 'POST',
        body: JSON.stringify({ flags: cleanedFlags }),
      });
      showNotification('Configuration updated', 'success');
      setIsEditing(false);
      setNewFlagSearch('');
      setShowFlagDropdown(false);
      onSetEditingConfig(experiment.id, false);
      // Wait for state update to propagate before refreshing
      // (loadData skips refresh when editingConfigs is not empty)
      setTimeout(() => onRefreshData(), 0);
    } catch (error) {
      console.error('Error updating configuration:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error updating configuration: ${msg}`, 'error');
    }
  };

  const addFlag = (flagDef: { flag: string; type: string; default: unknown; required: boolean }) => {
    // Add flag with default value to editedFlags
    setEditedFlags(prev => ({
      ...prev,
      [flagDef.flag]: flagDef.default !== null && flagDef.default !== undefined ? flagDef.default : '',
    }));
    setNewFlagSearch('');
    setShowFlagDropdown(false);
  };

  const deleteFlag = (flagName: string) => {
    setEditedFlags(prev => {
      const newFlags = { ...prev };
      delete newFlags[flagName];
      return newFlags;
    });
  };

  const fullCommand = buildCommand(experiment);

  return (
    <div className="expanded-details">
      <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start', marginBottom: '20px' }}>
        {/* Configuration Panel */}
        <div className="detail-section" style={{ width: 'auto', minWidth: '400px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              Configuration
              {isEditing ? (
                <span>
                  <a onClick={saveConfig} className="wandb-link" style={{ cursor: 'pointer', marginRight: '6px', fontSize: '12px', color: '#4caf50' }} title="Save changes">✓</a>
                  <a onClick={cancelEditing} className="wandb-link" style={{ cursor: 'pointer', fontSize: '12px', color: '#f44336' }} title="Cancel">✗</a>
                </span>
              ) : (
                <>
                  <a onClick={startEditing} className="wandb-link" style={{ cursor: 'pointer', fontSize: '11px' }} title="Edit configuration">✎</a>
                  <span
                    className="wandb-link"
                    onClick={handleCopy(`https://wandb.ai/metta-research/metta/runs/${experiment.name}`, 'W&B URL', `https://wandb.ai/metta-research/metta/runs/${experiment.name}`)}
                    title="Click to copy, Cmd+click to open"
                  >
                    W&B
                  </span>
                  <span
                    className="wandb-link"
                    onClick={handleCopy(`https://app.datadoghq.com/logs?query=metta_run_id%3A%22${experiment.name}%22`, 'log URL', `https://app.datadoghq.com/logs?query=metta_run_id%3A%22${experiment.name}%22`)}
                    title="Click to copy, Cmd+click to open"
                  >
                    log
                  </span>
                </>
              )}
            </h3>
          </div>
          <div className="detail-grid">
            <span className="detail-label">Name:</span>
            <span className="detail-value">
              {isEditing ? (
                <input
                  type="text"
                  value={editedConfig.name || ''}
                  onChange={e => setEditedConfig(prev => ({ ...prev, name: e.target.value }))}
                  style={{ width: '100%' }}
                />
              ) : (
                experiment.name
              )}
            </span>

            <span className="detail-label">Tool Path:</span>
            <span className="detail-value">
              {isEditing ? (
                <input
                  type="text"
                  value={editedConfig.tool_path || ''}
                  onChange={e => setEditedConfig(prev => ({ ...prev, tool_path: e.target.value }))}
                  style={{ width: '100%' }}
                />
              ) : (
                experiment.tool_path || '-'
              )}
            </span>

            <span className="detail-label">Nodes x GPUs:</span>
            <span className="detail-value">
              {isEditing ? (
                <>
                  <input
                    type="number"
                    value={editedConfig.nodes || 0}
                    onChange={e => setEditedConfig(prev => ({ ...prev, nodes: parseInt(e.target.value) }))}
                    style={{ width: '60px' }}
                  />
                  {' x '}
                  <input
                    type="number"
                    value={editedConfig.gpus || 0}
                    onChange={e => setEditedConfig(prev => ({ ...prev, gpus: parseInt(e.target.value) }))}
                    style={{ width: '60px' }}
                  />
                </>
              ) : (
                `${experiment.nodes} x ${experiment.gpus}`
              )}
            </span>

            <span className="detail-label">Git Branch:</span>
            <span className="detail-value">
              {isEditing ? (
                <input
                  type="text"
                  value={editedConfig.git_branch || ''}
                  onChange={e => setEditedConfig(prev => ({ ...prev, git_branch: e.target.value }))}
                  style={{ width: '100%' }}
                />
              ) : (
                experiment.git_branch || '-'
              )}
            </span>
          </div>

          {experiment.description && (
            <p style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>{experiment.description}</p>
          )}

          <hr style={{ margin: '15px 0', border: 'none', borderTop: '1px solid #ddd' }} />

          {/* Flags table */}
          <table style={{ width: 'auto', borderCollapse: 'collapse' }}>
            <tbody>
              {Object.entries(isEditing ? editedFlags : experiment.flags || {})
                .sort((a, b) => a[0].localeCompare(b[0]))
                .map(([key, val]) => (
                  <tr key={key} className="flag-row">
                    <td style={{ padding: '4px 8px 4px 0', fontSize: '12px', color: '#666', borderBottom: '1px solid #eee' }}>
                      {key}
                    </td>
                    <td style={{ padding: '4px 0 4px 12px', fontSize: '12px', borderBottom: '1px solid #eee' }}>
                      {isEditing ? (
                        <span
                          contentEditable
                          suppressContentEditableWarning
                          className="editable-value"
                          onBlur={e => {
                            const text = e.currentTarget.textContent || '';
                            let value: unknown = text;
                            try { value = JSON.parse(text); } catch { /* keep as string */ }
                            setEditedFlags(prev => ({ ...prev, [key]: value }));
                          }}
                        >
                          {String(editedFlags[key] ?? '')}
                        </span>
                      ) : typeof val === 'boolean' ? (
                        <span style={{
                          fontWeight: 800,
                          fontSize: '18px',
                          color: val ? '#2e7d32' : '#c62828',
                        }}>
                          {val ? '✓' : '✗'}
                        </span>
                      ) : String(val)}
                    </td>
                    {isEditing && (
                      <td style={{ padding: '4px 0 4px 8px', borderBottom: '1px solid #eee' }}>
                        <span
                          onClick={() => deleteFlag(key)}
                          style={{ cursor: 'pointer', color: '#f44336', fontSize: '14px' }}
                          title="Delete flag"
                        >
                          ✕
                        </span>
                      </td>
                    )}
                  </tr>
                ))}
              {Object.keys(isEditing ? editedFlags : experiment.flags || {}).length === 0 && (
                <tr><td colSpan={3} style={{ padding: '8px 0', fontSize: '12px', color: '#999' }}>No flags</td></tr>
              )}
              {isEditing && (
                <tr>
                  <td colSpan={3} style={{ padding: '8px 0', position: 'relative' }}>
                    <input
                      type="text"
                      placeholder="Type to search flags..."
                      value={newFlagSearch}
                      onChange={e => {
                        setNewFlagSearch(e.target.value);
                        setShowFlagDropdown(true);
                      }}
                      onFocus={() => setShowFlagDropdown(true)}
                      onClick={e => e.stopPropagation()}
                      onKeyDown={e => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          const trimmedValue = newFlagSearch.trim();
                          if (trimmedValue) {
                            // Check if this matches an existing flag definition
                            const matchingDef = flagDefinitions.find(f => f.flag === trimmedValue);
                            if (matchingDef) {
                              addFlag(matchingDef);
                            } else {
                              // Add custom flag with empty default
                              addFlag({ flag: trimmedValue, type: 'string', default: '', required: false });
                            }
                            setNewFlagSearch('');
                            setShowFlagDropdown(false);
                          }
                        } else if (e.key === 'Escape') {
                          e.preventDefault();
                          setNewFlagSearch('');
                          setShowFlagDropdown(false);
                        }
                      }}
                      style={{
                        width: '100%',
                        padding: '6px 8px',
                        fontSize: '12px',
                        border: '1px solid #ddd',
                        borderRadius: '3px',
                      }}
                    />
                    {showFlagDropdown && flagDefinitions.length > 0 && (
                      <div
                        onClick={e => e.stopPropagation()}
                        style={{
                          position: 'absolute',
                          top: '100%',
                          left: 0,
                          right: 0,
                          maxHeight: '300px',
                          overflowY: 'auto',
                          background: 'white',
                          border: '1px solid #ddd',
                          borderRadius: '3px',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                          zIndex: 1000,
                        }}
                      >
                        {flagDefinitions
                          .filter(f => {
                            // Filter out already added flags
                            if (editedFlags.hasOwnProperty(f.flag)) return false;
                            // Filter by search query
                            if (!newFlagSearch) return true;
                            return f.flag.toLowerCase().includes(newFlagSearch.toLowerCase());
                          })
                          .slice(0, 20) // Show max 20 results
                          .map(flagDef => (
                            <div
                              key={flagDef.flag}
                              onClick={() => addFlag(flagDef)}
                              style={{
                                padding: '8px 12px',
                                cursor: 'pointer',
                                borderBottom: '1px solid #eee',
                                fontSize: '12px',
                              }}
                              onMouseEnter={e => e.currentTarget.style.background = '#f5f5f5'}
                              onMouseLeave={e => e.currentTarget.style.background = 'white'}
                            >
                              <div style={{ fontWeight: 500 }}>{flagDef.flag}</div>
                              <div style={{ fontSize: '11px', color: '#666', marginTop: '2px' }}>
                                <span style={{ color: '#2196f3' }}>{flagDef.type}</span>
                                <span> • default: <code style={{ background: '#f5f5f5', padding: '1px 4px', borderRadius: '2px' }}>
                                  {flagDef.default === null ? 'null' : flagDef.default === undefined ? 'undefined' : String(flagDef.default)}
                                </code></span>
                              </div>
                            </div>
                          ))}
                        {flagDefinitions.filter(f =>
                          !editedFlags.hasOwnProperty(f.flag) &&
                          (!newFlagSearch || f.flag.toLowerCase().includes(newFlagSearch.toLowerCase()))
                        ).length === 0 && (
                          <div style={{ padding: '12px', fontSize: '12px', color: '#999', textAlign: 'center' }}>
                            No matching flags found
                          </div>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Jobs Panel */}
        <div className="detail-section" style={{ width: 'auto', flexShrink: 0 }}>
          <h3>
            Jobs
            {health?.skypilot && formatStaleness(health.skypilot.staleness_seconds) && (
              <span style={{ marginLeft: '8px', fontSize: '11px', color: '#999', fontWeight: 'normal' }}>
                (stale: {formatStaleness(health.skypilot.staleness_seconds)})
              </span>
            )}
          </h3>
          <ScrollPanel deps={[jobs]} maxHeight={300}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Job ID</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Status</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Age</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Duration</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Links</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(job => {
                  const numericId = job.id.split('-').pop();
                  return (
                    <tr key={job.id}>
                      <td
                        style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee', cursor: 'pointer' }}
                        onClick={handleCopy(job.id, 'job ID')}
                        title="Click to copy"
                      >
                        {numericId}
                      </td>
                      <td style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee' }}>
                        <span className={`status-badge ${job.status.toLowerCase()}`} title={job.status}>{abbreviateStatus(job.status)}</span>
                      </td>
                      <td style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee', color: '#666' }}>
                        {formatAge(job.started_at || job.created_at)}
                      </td>
                      <td style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee', color: '#666' }}>
                        {formatDuration(job.started_at, job.ended_at)}
                      </td>
                      <td style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee' }}>
                        <a href={`https://skypilot-api.softmax-research.net/dashboard/jobs/${job.id}`} target="_blank" className="wandb-link" rel="noreferrer">sky</a>
                        {' '}
                        <a href={`https://app.datadoghq.com/logs?query=skypilot_task_id%3A%2A${job.id}%2A%20metta_run_id%3A%22${job.experiment_id}%22`} target="_blank" className="wandb-link" rel="noreferrer">log</a>
                      </td>
                    </tr>
                  );
                })}
                {jobs.length === 0 && (
                  <tr><td colSpan={5} style={{ padding: '8px', color: '#999' }}>No jobs</td></tr>
                )}
              </tbody>
            </table>
          </ScrollPanel>
        </div>

        {/* Checkpoints Panel */}
        <div className="detail-section" style={{ flexShrink: 0 }}>
          <h3>
            Checkpoints
            {health?.s3 && formatStaleness(health.s3.staleness_seconds) && (
              <span style={{ marginLeft: '8px', fontSize: '11px', color: '#999', fontWeight: 'normal' }}>
                (stale: {formatStaleness(health.s3.staleness_seconds)})
              </span>
            )}
          </h3>
          <ScrollPanel deps={[checkpoints]} maxHeight={300}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Epoch</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Age</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Version</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Links</th>
                  <th style={{ padding: '4px 6px', textAlign: 'center', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Replays</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map(cp => (
                  <tr key={`${cp.epoch}-${cp.created_at}`}>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee' }}>{cp.epoch}</td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee', color: '#666' }}>{formatAge(cp.created_at)}</td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee', color: '#666' }}>{cp.policy_version || '-'}</td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee' }}>
                      {cp.model_path && (
                        <span className="storage-pill" style={{ backgroundColor: '#E8F5E9', color: '#2E7D32' }} onClick={handleCopy(cp.model_path!, 's3 path')}>s3</span>
                      )}
                      <span className="storage-pill" style={{ backgroundColor: '#E3F2FD', color: '#1565C0' }} onClick={handleCopy(cp.version ? `metta://policies/${cp.version}` : `metta://policy/${experiment.name}`, 'metta URI')}>metta</span>
                      <span className="storage-pill" style={{ backgroundColor: '#F3E5F5', color: '#6A1B9A' }} onClick={handleCopy(cp.observatory_url || `https://observatory.softmax-research.net/policy/${encodeURIComponent(experiment.name)}`, 'observatory URL', cp.observatory_url || `https://observatory.softmax-research.net/policy/${encodeURIComponent(experiment.name)}`)}>obs</span>
                    </td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee', textAlign: 'center' }}>{cp.replay_paths?.length || 0}</td>
                  </tr>
                ))}
                {checkpoints.length === 0 && (
                  <tr><td colSpan={5} style={{ padding: '8px', color: '#999' }}>No checkpoints yet</td></tr>
                )}
              </tbody>
            </table>
          </ScrollPanel>
        </div>
      </div>

      {/* Command Panel */}
      <div className="detail-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <h3 style={{ margin: 0 }}>Command</h3>
          <button className="copy-btn" onClick={handleCopy(fullCommand, 'command')} title="Copy command">⎘</button>
        </div>
        <div style={{ backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '12px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {fullCommand}
        </div>
      </div>
    </div>
  );
}
