import { useState, useMemo, useCallback } from 'react';
import type { Job } from '../types';
import { ScrollPanel } from './ScrollPanel';
import './JobsTable.css';

interface JobsTableProps {
  jobs: Job[];
  showStoppedJobs: boolean;
  showOrphanedOnly: boolean;
  showMyJobsOnly: boolean;
  jobsLimit: number;
  jobsFilterText: string;
  onToggleStoppedFilter: (checked: boolean) => void;
  onToggleOrphanedFilter: (checked: boolean) => void;
  onToggleMyJobsFilter: (checked: boolean) => void;
  onUpdateJobsLimit: (value: number) => void;
  onFilterJobs: (text: string) => void;
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
  return statusMap[status.toLowerCase()] || status.charAt(0).toUpperCase();
}

function formatTime(timestamp: string | null): string {
  if (!timestamp) return 'Not started';
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return date.toLocaleDateString();
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

function parseJobCommand(command: string) {
  const result: { branch: string | null; flags: Record<string, string>; fullCommand: string } = {
    branch: null,
    flags: {},
    fullCommand: command || '',
  };
  if (!command) return result;

  const branchMatch = command.match(/--(?:git[-_]?)?branch[=\s]+([^\s]+)/i) ||
                      command.match(/git checkout ([^\s;&|]+)/);
  if (branchMatch) result.branch = branchMatch[1];

  const flagMatches = [...command.matchAll(/--([a-zA-Z][-a-zA-Z0-9_]*?)(?:=|\s+)([^\s-][^\s]*?)(?=\s+--|$|\s+[^-])/g)];
  flagMatches.forEach(match => {
    result.flags[match[1]] = match[2];
  });

  return result;
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text);
}

export function JobsTable({
  jobs,
  showStoppedJobs,
  showOrphanedOnly,
  showMyJobsOnly,
  jobsLimit,
  jobsFilterText,
  onToggleStoppedFilter,
  onToggleOrphanedFilter,
  onToggleMyJobsFilter,
  onUpdateJobsLimit,
  onFilterJobs,
}: JobsTableProps) {
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set());
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set());

  const toggleJobExpanded = useCallback((jobId: string) => {
    setExpandedJobs(prev => {
      const next = new Set(prev);
      if (next.has(jobId)) {
        next.delete(jobId);
      } else {
        next.add(jobId);
      }
      return next;
    });
  }, []);

  const toggleJobSelection = useCallback((jobId: string, checked: boolean) => {
    setSelectedJobs(prev => {
      const next = new Set(prev);
      if (checked) {
        next.add(jobId);
      } else {
        next.delete(jobId);
      }
      return next;
    });
  }, []);

  // Parse resources from command
  const getResources = useCallback((job: Job) => {
    const command = job.command || '';
    const nodesMatch = command.match(/--nodes=(\d+)/);
    const gpusMatch = command.match(/--gpus=(\d+)/);
    if (nodesMatch && gpusMatch) {
      return `${nodesMatch[1]}×${gpusMatch[1]}`;
    }
    return `${job.nodes}×${job.gpus}`;
  }, []);

  const stoppableJobs = useMemo(() =>
    jobs.filter(j => j.status.toLowerCase() === 'running' || j.status.toLowerCase() === 'pending'),
    [jobs]
  );

  const allStoppableSelected = selectedJobs.size === stoppableJobs.length && stoppableJobs.length > 0;
  const someSelected = selectedJobs.size > 0 && selectedJobs.size < stoppableJobs.length;

  const toggleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedJobs(new Set(stoppableJobs.map(j => j.id)));
    } else {
      setSelectedJobs(new Set());
    }
  };

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h2 style={{ margin: 0 }}>SkyPilot Jobs</h2>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '13px' }}>
            Filter:
            <input
              type="text"
              placeholder="Search..."
              value={jobsFilterText}
              onChange={e => onFilterJobs(e.target.value)}
              style={{ width: '150px', padding: '4px' }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '13px' }}>
            <input
              type="checkbox"
              checked={showMyJobsOnly}
              onChange={e => onToggleMyJobsFilter(e.target.checked)}
            />
            My Jobs
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '13px' }}>
            <input
              type="checkbox"
              checked={showStoppedJobs}
              onChange={e => onToggleStoppedFilter(e.target.checked)}
            />
            Show Stopped
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '13px' }}>
            <input
              type="checkbox"
              checked={showOrphanedOnly}
              onChange={e => onToggleOrphanedFilter(e.target.checked)}
            />
            Orphaned Only
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '13px' }}>
            Limit:
            <input
              type="number"
              value={jobsLimit}
              min={1}
              max={1000}
              onChange={e => onUpdateJobsLimit(parseInt(e.target.value) || 20)}
              style={{ width: '60px', padding: '4px' }}
            />
          </label>
        </div>
      </div>
      <ScrollPanel className="table-wrapper" deps={[jobs]} maxHeight={400}>
        <table id="jobs-table">
          <thead>
            <tr>
              <th className="col-expand"></th>
              <th className="col-checkbox">
                <input
                  type="checkbox"
                  checked={allStoppableSelected}
                  ref={el => { if (el) el.indeterminate = someSelected; }}
                  onChange={e => toggleSelectAll(e.target.checked)}
                />
              </th>
              <th>Job ID</th>
              <th>Name</th>
              <th>Status</th>
              <th>Resources</th>
              <th>Duration</th>
              <th>Started</th>
              <th className="col-scrollbar"></th>
            </tr>
          </thead>
          <tbody id="jobs-tbody">
            {jobs.length === 0 ? (
              <tr>
                <td colSpan={9} className="empty-state">No jobs match the current filters</td>
              </tr>
            ) : (
              jobs.map(job => {
                const isExpanded = expandedJobs.has(job.id);
                const isSelected = selectedJobs.has(job.id);
                const canStop = job.status.toLowerCase() === 'running' || job.status.toLowerCase() === 'pending';
                const parsed = parseJobCommand(job.command);

                return [
                  <tr
                    key={job.id}
                    className="main-row"
                    data-job-id={job.id}
                    onClick={() => toggleJobExpanded(job.id)}
                  >
                    <td className="col-expand">
                      <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>▶</span>
                    </td>
                    <td className="col-checkbox" onClick={e => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        disabled={!canStop}
                        checked={isSelected}
                        onChange={e => toggleJobSelection(job.id, e.target.checked)}
                      />
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                      <span
                        onClick={e => { e.stopPropagation(); copyToClipboard(job.id); }}
                        style={{ cursor: 'pointer' }}
                        title="Click to copy"
                      >
                        {job.id}
                      </span>
                      <a
                        href={`https://wandb.ai/metta-research/metta/runs/${job.experiment_id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="wandb-link"
                        onClick={e => e.stopPropagation()}
                        style={{ marginLeft: '6px' }}
                      >
                        w&b
                      </a>
                      <a
                        href={`https://skypilot-api.softmax-research.net/dashboard/jobs/${job.id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="wandb-link"
                        onClick={e => e.stopPropagation()}
                        style={{ marginLeft: '6px' }}
                      >
                        sky
                      </a>
                      <a
                        href={`https://app.datadoghq.com/logs?query=skypilot_task_id%3A%2A${job.id}%2A%20metta_run_id%3A%22${job.experiment_id}%22`}
                        target="_blank"
                        rel="noreferrer"
                        className="wandb-link"
                        onClick={e => e.stopPropagation()}
                        style={{ marginLeft: '6px' }}
                      >
                        log
                      </a>
                    </td>
                    <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      <span
                        onClick={e => { e.stopPropagation(); copyToClipboard(job.experiment_id); }}
                        style={{ cursor: 'pointer' }}
                        title="Click to copy"
                      >
                        {job.experiment_id}
                      </span>
                    </td>
                    <td>
                      <span className={`status-badge ${job.status.toLowerCase()}`} title={job.status}>
                        {abbreviateStatus(job.status)}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'monospace', fontSize: '11px' }}>{getResources(job)}</td>
                    <td style={{ fontSize: '11px' }}>{formatDuration(job.started_at)}</td>
                    <td style={{ fontSize: '11px' }}>{formatTime(job.started_at)}</td>
                    <td></td>
                  </tr>,
                  <tr
                    key={`${job.id}-expanded`}
                    className={`expanded-row ${isExpanded ? 'show' : ''}`}
                    data-job-id={job.id}
                  >
                    <td colSpan={9} style={{ padding: '12px 20px', background: '#f9f9f9', borderTop: 'none' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
                        <div>
                          {parsed.branch && (
                            <div style={{ marginBottom: '12px' }}>
                              <strong style={{ display: 'block', marginBottom: '4px' }}>Branch:</strong>
                              <span style={{ fontFamily: 'monospace', padding: '2px 6px', background: '#e8f4f8', borderRadius: '3px' }}>
                                {parsed.branch}
                              </span>
                            </div>
                          )}
                          <div>
                            <strong style={{ display: 'block', marginBottom: '4px' }}>Flags:</strong>
                            {Object.keys(parsed.flags).length > 0 ? (
                              Object.entries(parsed.flags).map(([key, val]) => (
                                <div key={key} style={{ display: 'flex', gap: '8px', padding: '4px 0' }}>
                                  <span style={{ color: '#666', minWidth: '120px' }}>{key}:</span>
                                  <span style={{ fontFamily: 'monospace' }}>{val}</span>
                                </div>
                              ))
                            ) : (
                              <div style={{ color: '#999' }}>No flags detected</div>
                            )}
                          </div>
                        </div>
                        <div>
                          <strong style={{ display: 'block', marginBottom: '4px' }}>Command:</strong>
                          <div
                            style={{
                              background: 'white',
                              padding: '8px',
                              border: '1px solid #ddd',
                              borderRadius: '4px',
                              fontFamily: 'monospace',
                              fontSize: '11px',
                              wordBreak: 'break-all',
                              cursor: 'pointer',
                            }}
                            onClick={() => copyToClipboard(parsed.fullCommand)}
                            title="Click to copy"
                          >
                            {parsed.fullCommand}
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>,
                ];
              })
            )}
          </tbody>
        </table>
      </ScrollPanel>
    </>
  );
}
