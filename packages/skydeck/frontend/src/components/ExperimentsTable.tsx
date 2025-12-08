import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { Experiment, Checkpoint } from '../types';
import { useApi } from '../hooks/useApi';
import { useNotifications } from '../hooks/useNotifications';
import { ScrollPanel } from './ScrollPanel';
import './ExperimentsTable.css';

interface ExperimentsTableProps {
  experiments: Experiment[];
  expandedExperiments: Set<string>;
  selectedExperiments: Set<string>;
  editingConfigs: Set<string>;
  onToggleExpanded: (id: string) => void;
  onToggleStar: (id: string) => void;
  onSetDesiredState: (id: string, state: 'RUNNING' | 'STOPPED') => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
  onQuickCreate: () => void;
  onReorder: (order: string[]) => void;
  onSelectExperiment: (id: string, selected: boolean) => void;
  onSelectAll: (selectAll: boolean) => void;
  onBulkStart: () => void;
  onBulkStop: () => void;
  onBulkDuplicate: () => void;
  onBulkDelete: () => void;
  onSetEditingConfig: (id: string, editing: boolean) => void;
  onRefreshData: () => void;
}

// Utility functions
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

function formatLargeNumber(value: number): string {
  const absValue = Math.abs(value);
  if (absValue >= 1_000_000_000) {
    const formatted = (value / 1_000_000_000).toFixed(1);
    return formatted.endsWith('.0') ? `${formatted.slice(0, -2)}B` : `${formatted}B`;
  }
  if (absValue >= 1_000_000) {
    const formatted = (value / 1_000_000).toFixed(1);
    return formatted.endsWith('.0') ? `${formatted.slice(0, -2)}M` : `${formatted}M`;
  }
  if (absValue >= 1_000) {
    const formatted = (value / 1_000).toFixed(1);
    return formatted.endsWith('.0') ? `${formatted.slice(0, -2)}K` : `${formatted}K`;
  }
  return String(value);
}

function parseLargeNumber(input: string): number | string {
  // Remove underscores
  let cleaned = input.replace(/_/g, '');

  // Check for suffix (K, M, B)
  const match = cleaned.match(/^(-?[\d.]+)([KMB])$/i);
  if (match) {
    const num = parseFloat(match[1]);
    const suffix = match[2].toUpperCase();
    const multiplier = suffix === 'K' ? 1_000 : suffix === 'M' ? 1_000_000 : 1_000_000_000;
    return Math.round(num * multiplier);
  }

  // Try parsing as regular number
  const parsed = Number(cleaned);
  if (!isNaN(parsed)) {
    return parsed;
  }

  // Return as-is if not parseable as number
  return input;
}

function formatFlagValue(value: unknown): React.ReactElement {
  if (value === undefined || value === null) {
    return <span className="flag-value empty">-</span>;
  }
  if (typeof value === 'boolean') {
    const className = value ? 'boolean-true' : 'boolean-false';
    const emoji = value ? '✓' : '✗';
    return <span className={`flag-value ${className}`} style={{ fontSize: '200%', display: 'block', textAlign: 'center' }}>{emoji}</span>;
  }
  if (typeof value === 'number' && Number.isInteger(value)) {
    return <span className="flag-value">{formatLargeNumber(value)}</span>;
  }
  return <span className="flag-value">{String(value)}</span>;
}

function formatDuration(startedAt: string | null): string {
  if (!startedAt) return '-';
  const start = new Date(startedAt);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();
  const hours = Math.floor(diffMs / 3600000);
  const mins = Math.floor((diffMs % 3600000) / 60000);
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
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

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text);
}

// Build hierarchical flag headers
function buildFlagTree(flagColumns: string[]) {
  const root: { children: Record<string, TreeNode> } = { children: {} };
  let maxDepth = 0;

  interface TreeNode {
    label: string;
    fullPath: string;
    children: Record<string, TreeNode>;
    isLeaf: boolean;
  }

  flagColumns.forEach(flag => {
    const parts = flag.split('.');
    maxDepth = Math.max(maxDepth, parts.length);
    let current: { children: Record<string, TreeNode> } = root;
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (!current.children[part]) {
        current.children[part] = {
          label: part,
          fullPath: parts.slice(0, i + 1).join('.'),
          children: {},
          isLeaf: i === parts.length - 1,
        };
      }
      current = current.children[part];
    }
  });

  return { root, maxDepth };
}

function countLeaves(node: { children: Record<string, unknown> }): number {
  if (Object.keys(node.children).length === 0) return 1;
  return Object.values(node.children).reduce<number>((sum, child) => sum + countLeaves(child as { children: Record<string, unknown> }), 0);
}

function getLeafColumnsFromTree(tree: { children: Record<string, unknown> }): string[] {
  const leaves: string[] = [];
  interface TreeNode { label: string; fullPath: string; children: Record<string, TreeNode> }

  function traverse(node: TreeNode) {
    if (Object.keys(node.children).length === 0) {
      leaves.push(node.fullPath);
    } else {
      Object.values(node.children)
        .sort((a, b) => a.label.localeCompare(b.label))
        .forEach(child => traverse(child));
    }
  }

  Object.values(tree.children as Record<string, TreeNode>)
    .sort((a, b) => a.label.localeCompare(b.label))
    .forEach(node => traverse(node));

  return leaves;
}

// StateTransition component
function StateTransition({ current, desired, experimentId, onSetDesiredState }: {
  current: string;
  desired: string;
  experimentId: string;
  onSetDesiredState: (id: string, state: 'RUNNING' | 'STOPPED') => void;
}) {
  const [editing, setEditing] = useState(false);
  const currentLower = current.toLowerCase();
  const desiredLower = desired.toLowerCase();

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditing(!editing);
  };

  const handleStateChange = (state: 'RUNNING' | 'STOPPED', e: React.MouseEvent) => {
    e.stopPropagation();
    onSetDesiredState(experimentId, state);
    setEditing(false);
  };

  if (editing) {
    return (
      <span style={{ display: 'inline-flex', gap: '2px', whiteSpace: 'nowrap' }}>
        <span
          className="state-pill stopped"
          onClick={(e) => handleStateChange('STOPPED', e)}
          title="Stop"
          style={{ cursor: 'pointer', padding: '2px 6px', borderRadius: '8px', backgroundColor: '#9E9E9E', color: 'white', fontSize: '12px' }}
        >
          ■
        </span>
        <span style={{ color: '#666', fontSize: '12px' }}>|</span>
        <span
          className="state-pill running"
          onClick={(e) => handleStateChange('RUNNING', e)}
          title="Start"
          style={{ cursor: 'pointer', padding: '2px 6px', borderRadius: '8px', backgroundColor: '#4CAF50', color: 'white', fontSize: '12px' }}
        >
          ▶
        </span>
      </span>
    );
  }

  const currentAbbrev = abbreviateStatus(currentLower);
  const desiredAbbrev = abbreviateStatus(desiredLower);

  if (currentLower === desiredLower) {
    return (
      <span onClick={handleClick} style={{ cursor: 'pointer' }}>
        <span className={`status-badge ${currentLower}`} title={current}>{currentAbbrev}</span>
      </span>
    );
  }

  if (currentLower === 'failed' && desiredLower === 'stopped') {
    return (
      <span onClick={handleClick} style={{ cursor: 'pointer' }}>
        <span className={`status-badge ${currentLower}`} title={current}>{currentAbbrev}</span>
      </span>
    );
  }

  return (
    <span onClick={handleClick} style={{ cursor: 'pointer' }}>
      <span className={`status-badge ${currentLower}`} title={current}>{currentAbbrev}</span>
      {' → '}
      <span className={`status-badge ${desiredLower}`} title={desired}>{desiredAbbrev}</span>
    </span>
  );
}

// ExpandedDetails component
function ExpandedDetails({ experiment, onSetEditingConfig, onRefreshData }: {
  experiment: Experiment;
  onSetEditingConfig: (id: string, editing: boolean) => void;
  onRefreshData: () => void;
}) {
  const { apiCall } = useApi();
  const { showNotification } = useNotifications();
  const [jobs, setJobs] = useState<Array<{ id: string; status: string; experiment_id: string }>>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [editedConfig, setEditedConfig] = useState<Record<string, string | number | null>>({});
  const [editedFlags, setEditedFlags] = useState<Record<string, unknown>>({});
  const [flagDefinitions, setFlagDefinitions] = useState<Array<{ flag: string; type: string; default: unknown; required: boolean }>>([]);
  const [newFlagSearch, setNewFlagSearch] = useState('');
  const [showFlagDropdown, setShowFlagDropdown] = useState(false);
  const [editingBooleanFlag, setEditingBooleanFlag] = useState<string | null>(null);

  // Load jobs and checkpoints
  useEffect(() => {
    const loadData = async () => {
      try {
        const [jobsData, checkpointsData] = await Promise.all([
          apiCall<{ jobs: Array<{ id: string; status: string; experiment_id: string }> }>(`/experiments/${experiment.id}/jobs?limit=10`),
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
      console.log('Loading flags with params:', params);
      const response = await apiCall<{ flags: Array<{ flag: string; type: string; default: unknown; required: boolean }> }>(`/flags${params}`);
      console.log('Loaded flag definitions:', response.flags?.length);
      setFlagDefinitions(response.flags || []);
    } catch (error) {
      console.error('Error loading flag definitions:', error);
      setFlagDefinitions([]);
    }
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setNewFlagSearch('');
    setShowFlagDropdown(false);
    onSetEditingConfig(experiment.id, false);
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

  const saveConfig = async () => {
    try {
      await apiCall(`/experiments/${experiment.id}`, {
        method: 'PATCH',
        body: JSON.stringify(editedConfig),
      });
      await apiCall(`/experiments/${experiment.id}/flags`, {
        method: 'POST',
        body: JSON.stringify({ flags: editedFlags }),
      });
      showNotification('Configuration updated', 'success');
      setIsEditing(false);
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
                <a onClick={startEditing} className="wandb-link" style={{ cursor: 'pointer', fontSize: '11px' }} title="Edit configuration">✎</a>
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
                <>
                  {experiment.name}
                  <a href={`https://wandb.ai/metta-research/metta/runs/${experiment.id}`} target="_blank" className="wandb-link" rel="noreferrer">W&B</a>
                  <a href={`https://app.datadoghq.com/logs?query=metta_run_id%3A%22${experiment.id}%22`} target="_blank" className="wandb-link" rel="noreferrer">log</a>
                </>
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
              {Object.entries(isEditing ? editedFlags : experiment.flags)
                .sort((a, b) => a[0].localeCompare(b[0]))
                .map(([key, val]) => (
                  <tr key={key} className="flag-row">
                    <td style={{ padding: '4px 8px 4px 0', fontSize: '12px', color: '#666', borderBottom: '1px solid #eee' }}>
                      {key}
                      {isEditing && (
                        <button
                          onClick={() => deleteFlag(key)}
                          style={{
                            marginLeft: '8px',
                            padding: '0 4px',
                            fontSize: '10px',
                            cursor: 'pointer',
                            background: '#f44336',
                            color: 'white',
                            border: 'none',
                            borderRadius: '3px',
                          }}
                          title="Delete flag"
                        >
                          ×
                        </button>
                      )}
                    </td>
                    <td style={{ padding: '4px 0 4px 12px', fontSize: '12px', borderBottom: '1px solid #eee' }}>
                      {isEditing ? (
                        typeof val === 'boolean' ? (
                          <span
                            style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}
                            onMouseEnter={() => setEditingBooleanFlag(key)}
                            onMouseLeave={() => setEditingBooleanFlag(null)}
                          >
                            {editingBooleanFlag === key ? (
                              <>
                                <span
                                  onClick={() => {
                                    setEditedFlags(prev => ({ ...prev, [key]: false }));
                                    setEditingBooleanFlag(null);
                                  }}
                                  style={{
                                    fontSize: '16px',
                                    color: val === false ? '#c62828' : '#999',
                                    fontWeight: val === false ? 'bold' : 'normal',
                                    cursor: 'pointer',
                                  }}
                                  title="False"
                                >
                                  ✗
                                </span>
                                <span style={{ color: '#ddd' }}>|</span>
                                <span
                                  onClick={() => {
                                    setEditedFlags(prev => ({ ...prev, [key]: true }));
                                    setEditingBooleanFlag(null);
                                  }}
                                  style={{
                                    fontSize: '16px',
                                    color: val === true ? '#2e7d32' : '#999',
                                    fontWeight: val === true ? 'bold' : 'normal',
                                    cursor: 'pointer',
                                  }}
                                  title="True"
                                >
                                  ✓
                                </span>
                              </>
                            ) : (
                              <span style={{ fontSize: '16px', color: val ? '#2e7d32' : '#c62828', fontWeight: 'bold' }}>
                                {val ? '✓' : '✗'}
                              </span>
                            )}
                          </span>
                        ) : (
                          <span
                            contentEditable
                            suppressContentEditableWarning
                            className="editable-value"
                            onBlur={e => {
                              const text = e.currentTarget.textContent || '';
                              let value: unknown = text;
                              // First try to parse as large number (handles underscores and K/M/B suffixes)
                              const parsed = parseLargeNumber(text);
                              if (typeof parsed === 'number') {
                                value = parsed;
                              } else {
                                // Try JSON parse for booleans and other types
                                try { value = JSON.parse(text); } catch { /* keep as string */ }
                              }
                              setEditedFlags(prev => ({ ...prev, [key]: value }));
                            }}
                          >
                            {typeof val === 'number' && Number.isInteger(val) ? formatLargeNumber(val) : String(val ?? '')}
                          </span>
                        )
                      ) : (typeof val === 'number' && Number.isInteger(val) ? formatLargeNumber(val) : String(val))}
                    </td>
                  </tr>
                ))}
              {Object.keys(experiment.flags).length === 0 && !isEditing && (
                <tr><td colSpan={2} style={{ padding: '8px 0', fontSize: '12px', color: '#999' }}>No flags</td></tr>
              )}
              {isEditing && (
                <tr>
                  <td colSpan={2} style={{ padding: '8px 0', position: 'relative' }}>
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
          <h3>Jobs</h3>
          <ScrollPanel deps={[jobs]} maxHeight={300}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Job ID</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e0e0e0', fontSize: '10px' }}>Status</th>
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
                        onClick={() => copyToClipboard(job.id)}
                        title="Click to copy"
                      >
                        {numericId}
                      </td>
                      <td style={{ padding: '4px 6px', fontSize: '11px', borderBottom: '1px solid #eee' }}>
                        <span className={`status-badge ${job.status.toLowerCase()}`} title={job.status}>{abbreviateStatus(job.status)}</span>
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
                  <tr><td colSpan={3} style={{ padding: '8px', color: '#999' }}>No jobs</td></tr>
                )}
              </tbody>
            </table>
          </ScrollPanel>
        </div>

        {/* Checkpoints Panel */}
        <div className="detail-section" style={{ flexShrink: 0 }}>
          <h3>Checkpoints</h3>
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
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee', color: '#666' }}>{formatDuration(cp.created_at)}</td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee', color: '#666' }}>{cp.policy_version || '-'}</td>
                    <td style={{ padding: '4px 6px', borderBottom: '1px solid #eee' }}>
                      {cp.model_path && (
                        <span className="storage-pill" style={{ backgroundColor: '#E8F5E9', color: '#2E7D32' }} onClick={() => copyToClipboard(cp.model_path!)}>s3</span>
                      )}
                      <span className="storage-pill" style={{ backgroundColor: '#E3F2FD', color: '#1565C0' }} onClick={() => copyToClipboard(cp.version ? `metta://policies/${cp.version}` : `metta://policy/${experiment.id}`)}>metta</span>
                      <span className="storage-pill" style={{ backgroundColor: '#F3E5F5', color: '#6A1B9A' }} onClick={() => copyToClipboard(cp.observatory_url || `https://observatory.softmax-research.net/policy/${encodeURIComponent(experiment.name)}`)}>obs</span>
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
          <button className="copy-btn" onClick={() => copyToClipboard(fullCommand)} title="Copy command">⎘</button>
        </div>
        <div style={{ backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '12px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {fullCommand}
        </div>
      </div>
    </div>
  );
}

export function ExperimentsTable({
  experiments,
  expandedExperiments,
  selectedExperiments,
  editingConfigs: _editingConfigs,
  onToggleExpanded,
  onToggleStar,
  onSetDesiredState,
  onDelete: _onDelete,
  onDuplicate: _onDuplicate,
  onQuickCreate,
  onReorder,
  onSelectExperiment,
  onSelectAll,
  onBulkStart,
  onBulkStop,
  onBulkDuplicate,
  onBulkDelete,
  onSetEditingConfig,
  onRefreshData,
}: ExperimentsTableProps) {
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const tbodyRef = useRef<HTMLTableSectionElement>(null);

  // Collect all unique flags
  const allFlags = useMemo(() => {
    const flags = new Set<string>();
    experiments.forEach(exp => {
      Object.keys(exp.flags || {}).forEach(flag => flags.add(flag));
    });
    return Array.from(flags).sort();
  }, [experiments]);

  // Build flag tree for headers
  const { root: flagTree, maxDepth } = useMemo(() => buildFlagTree(allFlags), [allFlags]);
  const flagColumns = useMemo(() => getLeafColumnsFromTree(flagTree), [flagTree]);

  // Handle drag and drop
  const handleDragStart = useCallback((id: string) => {
    setDraggedId(id);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedId(null);
  }, []);

  const handleDrop = useCallback((targetId: string) => {
    if (!draggedId || draggedId === targetId) return;

    const currentOrder = experiments.map(e => e.id);
    const draggedIndex = currentOrder.indexOf(draggedId);
    const targetIndex = currentOrder.indexOf(targetId);

    const newOrder = [...currentOrder];
    newOrder.splice(draggedIndex, 1);
    newOrder.splice(targetIndex, 0, draggedId);

    onReorder(newOrder);
    setDraggedId(null);
  }, [draggedId, experiments, onReorder]);

  // Render header rows
  const renderHeaderRows = () => {
    const totalRows = maxDepth || 1;
    const allSelected = selectedExperiments.size === experiments.length && experiments.length > 0;
    const someSelected = selectedExperiments.size > 0 && selectedExperiments.size < experiments.length;

    interface TreeNode { label: string; fullPath: string; children: Record<string, TreeNode> }

    const rows: React.ReactElement[][] = Array.from({ length: totalRows }, () => []);

    const renderLevel = (nodes: TreeNode[], depth: number) => {
      nodes.forEach(node => {
        const childCount = Object.keys(node.children).length;
        if (childCount === 0) {
          const rowspan = totalRows - depth;
          const displayName = node.label.replace(/\.enabled$/, '');
          rows[depth].push(
            <th key={node.fullPath} className="col-flag" rowSpan={rowspan} title={node.fullPath}>
              {displayName}
            </th>
          );
        } else {
          const colspan = countLeaves(node);
          const displayName = node.label.replace(/\.enabled$/, '');
          rows[depth].push(
            <th key={node.fullPath} className="col-flag-group" colSpan={colspan} title={node.fullPath}>
              {displayName}
            </th>
          );
          renderLevel(
            Object.values(node.children).sort((a, b) => a.label.localeCompare(b.label)),
            depth + 1
          );
        }
      });
    };

    if (allFlags.length > 0) {
      renderLevel(
        Object.values(flagTree.children as Record<string, TreeNode>).sort((a, b) => a.label.localeCompare(b.label)),
        0
      );
    }

    return (
      <>
        <tr>
          <th className="col-drag" rowSpan={totalRows} title="Drag to reorder">⋮⋮</th>
          <th className="col-checkbox" rowSpan={totalRows}>
            <input
              type="checkbox"
              checked={allSelected}
              ref={el => { if (el) el.indeterminate = someSelected; }}
              onChange={e => onSelectAll(e.target.checked)}
            />
          </th>
          <th className="col-expand" rowSpan={totalRows}>
            <span
              className="create-btn"
              onClick={onQuickCreate}
              title="Create new experiment"
            >
              +
            </span>
          </th>
          <th className="col-id" rowSpan={totalRows}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Name</span>
              {selectedExperiments.size > 0 && (
                <div className="bulk-actions">
                  <button onClick={onBulkStart} className="btn-pill btn-pill-primary">Start</button>
                  <button onClick={onBulkStop} className="btn-pill">Stop</button>
                  <button onClick={onBulkDuplicate} className="btn-pill">Duplicate</button>
                  <button onClick={onBulkDelete} className="btn-pill btn-pill-danger">Delete</button>
                </div>
              )}
            </div>
          </th>
          <th className="col-state" rowSpan={totalRows}>State</th>
          <th className="col-epoch" rowSpan={totalRows}>Epoch</th>
          {rows[0]}
          <th className="col-resources" rowSpan={totalRows}>Resources</th>
          <th className="col-padding" rowSpan={totalRows}></th>
        </tr>
        {rows.slice(1).map((row, i) => (
          <tr key={i + 1}>{row}</tr>
        ))}
      </>
    );
  };

  if (experiments.length === 0) {
    return (
      <div className="table-wrapper">
        <table id="experiments-table">
          <thead>{renderHeaderRows()}</thead>
          <tbody>
            <tr>
              <td colSpan={100} className="empty-state">
                No experiments found. Click the "+" button to create one.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table id="experiments-table">
        <thead id="experiments-thead">{renderHeaderRows()}</thead>
        <tbody ref={tbodyRef} id="experiments-tbody">
          {experiments.map(exp => {
            const expIdStr = String(exp.id);
            const isExpanded = expandedExperiments.has(expIdStr);
            const isSelected = selectedExperiments.has(expIdStr);
            const isDragging = draggedId === expIdStr;

            return [
              <tr
                key={exp.id}
                className={`main-row ${isDragging ? 'dragging' : ''}`}
                data-exp-id={exp.id}
                draggable
                onDragStart={() => handleDragStart(expIdStr)}
                onDragEnd={handleDragEnd}
                onDragOver={e => e.preventDefault()}
                onDrop={() => handleDrop(expIdStr)}
                onClick={() => onToggleExpanded(expIdStr)}
              >
                <td className="col-drag drag-handle">⋮⋮</td>
                <td className="col-checkbox" onClick={e => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={e => onSelectExperiment(expIdStr, e.target.checked)}
                  />
                </td>
                <td className="col-expand">
                  <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>▶</span>
                </td>
                <td className="col-id">
                  <button
                    className={`star-btn ${exp.starred ? 'starred' : ''}`}
                    onClick={e => { e.stopPropagation(); onToggleStar(expIdStr); }}
                    title={exp.starred ? 'Unstar' : 'Star'}
                  >
                    ★
                  </button>
                  <span
                    className="wandb-link"
                    onClick={e => { e.stopPropagation(); copyToClipboard(`https://wandb.ai/metta-research/metta/runs/${exp.id}`); }}
                    title="Click to copy W&B URL"
                  >
                    W&B
                  </span>
                  <a
                    href={`https://app.datadoghq.com/logs?query=metta_run_id%3A%22${exp.id}%22`}
                    target="_blank"
                    rel="noreferrer"
                    className="wandb-link"
                    onClick={e => e.stopPropagation()}
                  >
                    log
                  </a>
                  <span
                    onClick={e => { e.stopPropagation(); copyToClipboard(exp.name); }}
                    style={{ cursor: 'pointer' }}
                    title="Click to copy name"
                  >
                    {exp.name}
                  </span>
                </td>
                <td className="col-state">
                  <StateTransition
                    current={exp.current_state}
                    desired={exp.desired_state}
                    experimentId={exp.id}
                    onSetDesiredState={onSetDesiredState}
                  />
                </td>
                <td className="col-epoch">{exp.latest_epoch ?? '—'}</td>
                {flagColumns.map(flag => (
                  <td key={flag} className="col-flag">
                    {formatFlagValue(exp.flags[flag])}
                  </td>
                ))}
                <td className="col-resources">{exp.nodes}×{exp.gpus}</td>
                <td className="col-padding"></td>
              </tr>,
              <tr
                key={`${exp.id}-expanded`}
                className={`expanded-row ${isExpanded ? 'show' : ''}`}
                data-exp-id={exp.id}
              >
                <td colSpan={8 + flagColumns.length}>
                  {isExpanded && (
                    <ExpandedDetails
                      experiment={exp}
                      onSetEditingConfig={onSetEditingConfig}
                      onRefreshData={onRefreshData}
                    />
                  )}
                </td>
              </tr>,
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}
