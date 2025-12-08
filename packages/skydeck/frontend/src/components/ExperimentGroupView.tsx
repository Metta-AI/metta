import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { Experiment, ExperimentGroup } from '../types';
import { useApi } from '../hooks/useApi';
import { useNotifications } from '../hooks/useNotifications';
import './ExperimentsTable.css';

interface ExperimentGroupViewProps {
  group: ExperimentGroup | null;  // null = ungrouped
  experiments: Experiment[];
  expandedExperiments: Set<string>;
  selectedExperiments: Set<string>;
  allGroups: ExperimentGroup[];
  onToggleExpanded: (id: string) => void;
  onToggleStar: (id: string) => void;
  onSetDesiredState: (id: string, state: 'RUNNING' | 'STOPPED') => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
  onQuickCreate: (options?: { flags?: Record<string, unknown>; nodes?: number; gpus?: number; groupId?: string }) => void;
  onSelectExperiment: (id: string, selected: boolean) => void;
  onSelectAll: (selectAll: boolean, groupId: string | null) => void;
  onBulkStart: () => void;
  onBulkStop: () => void;
  onBulkDuplicate: () => void;
  onBulkDelete: () => void;
  onSetEditingConfig: (id: string, editing: boolean) => void;
  onRefreshData: () => void;
  onDragExperiment: (experimentId: string, targetGroupId: string | null, multiHome: boolean) => void;
  onReorderInGroup: (groupId: string | null, order: string[]) => void;
  onUpdateGroup: (groupId: string, updates: { name?: string; collapsed?: boolean }) => void;
  onDeleteGroup: (groupId: string) => void;
  onCreateGroup: () => void;
  // For rendering expanded details
  renderExpandedDetails: (experiment: Experiment) => React.ReactNode;
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

export function ExperimentGroupView({
  group,
  experiments,
  expandedExperiments,
  selectedExperiments,
  allGroups: _allGroups,
  onToggleExpanded,
  onToggleStar,
  onSetDesiredState,
  onDelete: _onDelete,
  onDuplicate: _onDuplicate,
  onQuickCreate,
  onSelectExperiment,
  onSelectAll,
  onBulkStart,
  onBulkStop,
  onBulkDuplicate,
  onBulkDelete,
  onSetEditingConfig: _onSetEditingConfig,
  onRefreshData,
  onDragExperiment,
  onReorderInGroup,
  onUpdateGroup,
  onDeleteGroup,
  onCreateGroup,
  renderExpandedDetails,
}: ExperimentGroupViewProps) {
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [insertionIndex, setInsertionIndex] = useState<number | null>(null);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(group?.name || '');
  const [editingFlag, setEditingFlag] = useState<string | null>(null);
  const [editedFlagValue, setEditedFlagValue] = useState<string>('');
  const [isAddingFlag, setIsAddingFlag] = useState(false);
  const [newFlagKey, setNewFlagKey] = useState('');
  const [newFlagValue, setNewFlagValue] = useState('');
  const [showFlagDropdown, setShowFlagDropdown] = useState(false);
  const [flagDefinitions, setFlagDefinitions] = useState<Array<{ flag: string; type: string; default: unknown; required: boolean }>>([]);
  const [editingResource, setEditingResource] = useState<'nodes' | 'gpus' | null>(null);
  const [editedResourceValue, setEditedResourceValue] = useState<string>('');
  const [isEditingToolPath, setIsEditingToolPath] = useState(false);
  const [editedToolPath, setEditedToolPath] = useState<string>('');
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const tbodyRef = useRef<HTMLTableSectionElement>(null);
  const dragCountRef = useRef(0);  // Track enter/leave balance
  const groupId = group?.id || null;
  const isUngrouped = !group;
  const { apiCall } = useApi();
  const { showNotification } = useNotifications();

  // Update a flag across all experiments in the group
  const updateFlagForAll = useCallback(async (flagKey: string, value: unknown) => {
    try {
      await Promise.all(experiments.map(exp =>
        apiCall(`/experiments/${exp.id}/flags`, {
          method: 'POST',
          body: JSON.stringify({ flags: { ...exp.flags, [flagKey]: value } }),
        })
      ));
      showNotification(`Updated ${flagKey} for ${experiments.length} experiments`, 'success');
      onRefreshData();
    } catch (error) {
      console.error('Error updating flag:', error);
      showNotification('Error updating flag', 'error');
    }
  }, [experiments, apiCall, showNotification, onRefreshData]);

  // Handle saving edited flag value
  const handleSaveFlagValue = useCallback(async (flagKey: string, valueStr: string) => {
    let value: unknown = valueStr;
    // First try to parse as large number (handles underscores and K/M/B suffixes)
    const parsed = parseLargeNumber(valueStr);
    if (typeof parsed === 'number') {
      value = parsed;
    } else {
      // Try JSON parse for booleans and other types
      try { value = JSON.parse(valueStr); } catch { /* keep as string */ }
    }
    await updateFlagForAll(flagKey, value);
    setEditingFlag(null);
    setEditedFlagValue('');
  }, [updateFlagForAll]);

  // Load flag definitions when starting to add a flag
  useEffect(() => {
    if (isAddingFlag && flagDefinitions.length === 0 && experiments.length > 0) {
      const loadFlags = async () => {
        try {
          const toolPath = experiments[0]?.tool_path || '';
          const params = toolPath ? `?tool_path=${encodeURIComponent(toolPath)}` : '';
          console.log('[GroupView] Loading flags with params:', params);
          const response = await apiCall<{ flags: Array<{ flag: string; type: string; default: unknown; required: boolean }> }>(`/flags${params}`);
          console.log('[GroupView] Loaded flag definitions:', response.flags?.length);
          setFlagDefinitions(response.flags || []);
        } catch (error) {
          console.error('[GroupView] Error loading flag definitions:', error);
        }
      };
      loadFlags();
    }
  }, [isAddingFlag, flagDefinitions.length, experiments, apiCall]);

  // Handle adding a new flag
  const handleAddFlag = useCallback(async (flagDef?: { flag: string; type: string; default: unknown; required: boolean }) => {
    let flagKey: string;
    let flagValue: unknown;

    if (flagDef) {
      flagKey = flagDef.flag;
      flagValue = flagDef.default !== null && flagDef.default !== undefined ? flagDef.default : '';
    } else {
      if (!newFlagKey.trim()) {
        showNotification('Flag key is required', 'error');
        return;
      }
      flagKey = newFlagKey.trim();
      flagValue = newFlagValue;
      try { flagValue = JSON.parse(newFlagValue); } catch { /* keep as string */ }
    }

    await updateFlagForAll(flagKey, flagValue);
    setIsAddingFlag(false);
    setNewFlagKey('');
    setNewFlagValue('');
    setShowFlagDropdown(false);
  }, [newFlagKey, newFlagValue, updateFlagForAll, showNotification]);

  // Update nodes or gpus across all experiments in the group
  const updateResourceForAll = useCallback(async (field: 'nodes' | 'gpus', value: number) => {
    try {
      await Promise.all(experiments.map(exp =>
        apiCall(`/experiments/${exp.id}`, {
          method: 'PATCH',
          body: JSON.stringify({ [field]: value }),
        })
      ));
      showNotification(`Updated ${field} for ${experiments.length} experiments`, 'success');
      onRefreshData();
    } catch (error) {
      console.error('Error updating resource:', error);
      showNotification('Error updating resource', 'error');
    }
  }, [experiments, apiCall, showNotification, onRefreshData]);

  // Update tool_path across all experiments in the group
  const updateToolPathForAll = useCallback(async (value: string) => {
    try {
      await Promise.all(experiments.map(exp =>
        apiCall(`/experiments/${exp.id}`, {
          method: 'PATCH',
          body: JSON.stringify({ tool_path: value || null }),
        })
      ));
      showNotification(`Updated tool_path for ${experiments.length} experiments`, 'success');
      onRefreshData();
    } catch (error) {
      console.error('Error updating tool_path:', error);
      showNotification('Error updating tool_path', 'error');
    }
  }, [experiments, apiCall, showNotification, onRefreshData]);

  // Delete a flag from all experiments in the group
  const deleteFlagForAll = useCallback(async (flagKey: string) => {
    try {
      await Promise.all(experiments.map(exp => {
        const newFlags = { ...exp.flags };
        delete newFlags[flagKey];
        return apiCall(`/experiments/${exp.id}/flags`, {
          method: 'POST',
          body: JSON.stringify({ flags: newFlags }),
        });
      }));
      showNotification(`Deleted ${flagKey} from ${experiments.length} experiments`, 'success');
      onRefreshData();
    } catch (error) {
      console.error('Error deleting flag:', error);
      showNotification('Error deleting flag', 'error');
    }
  }, [experiments, apiCall, showNotification, onRefreshData]);

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

  // Collect all unique flags from experiments in this group
  const allFlags = useMemo(() => {
    // If group has explicit flags, use those; otherwise derive from experiments
    if (group?.flags && group.flags.length > 0) {
      return group.flags;
    }
    const flags = new Set<string>();
    experiments.forEach(exp => {
      Object.keys(exp.flags || {}).forEach(flag => flags.add(flag));
    });
    return Array.from(flags).sort();
  }, [experiments, group?.flags]);

  // Find flags that are common across all experiments in the group (same value everywhere)
  // Also includes nodes, gpus, and tool_path as pseudo-flags
  // For single-experiment groups, treat all flags as common
  const { commonFlags, commonNodes, commonGpus, commonToolPath } = useMemo(() => {
    if (experiments.length === 0) return { commonFlags: {}, commonNodes: null, commonGpus: null, commonToolPath: null };

    // For single experiment, treat all flags as common (show in header)
    if (experiments.length === 1) {
      const exp = experiments[0];
      const common: Record<string, unknown> = {};
      for (const flag of allFlags) {
        if (exp.flags?.[flag] !== undefined) {
          common[flag] = exp.flags[flag];
        }
      }
      return { commonFlags: common, commonNodes: exp.nodes, commonGpus: exp.gpus, commonToolPath: exp.tool_path };
    }

    const common: Record<string, unknown> = {};

    // For each flag, check if all experiments have the same value
    for (const flag of allFlags) {
      const values = experiments.map(exp => exp.flags?.[flag]);
      const firstValue = values[0];

      // Check if all values are the same (use JSON.stringify for deep comparison)
      const firstStr = JSON.stringify(firstValue);
      const allSame = values.every(v => JSON.stringify(v) === firstStr);

      if (allSame && firstValue !== undefined) {
        common[flag] = firstValue;
      }
    }

    // Check if tool_path is common
    const toolPathValues = experiments.map(exp => exp.tool_path);
    const firstToolPath = toolPathValues[0];
    const toolPathCommon = toolPathValues.every(v => v === firstToolPath) ? firstToolPath : null;

    // Check if nodes are common
    const nodeValues = experiments.map(exp => exp.nodes);
    const firstNodes = nodeValues[0];
    const nodesCommon = nodeValues.every(v => v === firstNodes) ? firstNodes : null;

    // Check if gpus are common
    const gpuValues = experiments.map(exp => exp.gpus);
    const firstGpus = gpuValues[0];
    const gpusCommon = gpuValues.every(v => v === firstGpus) ? firstGpus : null;

    return { commonFlags: common, commonNodes: nodesCommon, commonGpus: gpusCommon, commonToolPath: toolPathCommon };
  }, [experiments, allFlags]);

  // Flags to show in table rows (excluding common flags)
  const rowFlags = useMemo(() => {
    return allFlags.filter(flag => !(flag in commonFlags));
  }, [allFlags, commonFlags]);

  // Build flag tree for headers (using rowFlags to exclude common flags)
  const { root: flagTree, maxDepth } = useMemo(() => buildFlagTree(rowFlags), [rowFlags]);
  const flagColumns = useMemo(() => getLeafColumnsFromTree(flagTree), [flagTree]);

  // Handle drag and drop within group
  const handleDragStart = useCallback((id: string, e: React.DragEvent) => {
    setDraggedId(id);
    e.dataTransfer.setData('text/plain', id);
    e.dataTransfer.setData('application/x-source-group', groupId || 'ungrouped');
    e.dataTransfer.effectAllowed = 'copyMove';
  }, [groupId]);

  const handleDragEnd = useCallback(() => {
    setDraggedId(null);
    setInsertionIndex(null);
  }, []);

  const handleRowDragOver = useCallback((targetExpId: string, e: React.DragEvent) => {
    e.preventDefault();
    const targetIndex = experiments.findIndex(exp => String(exp.id) === targetExpId);
    if (targetIndex !== -1) {
      // Determine if we should insert before or after based on mouse position
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const midY = rect.top + rect.height / 2;
      const insertBefore = e.clientY < midY;
      setInsertionIndex(insertBefore ? targetIndex : targetIndex + 1);
    }
  }, [experiments]);

  const handleDrop = useCallback((targetExpId: string, e: React.DragEvent) => {
    e.preventDefault();
    const draggedExpId = e.dataTransfer.getData('text/plain');
    const sourceGroupId = e.dataTransfer.getData('application/x-source-group');
    const multiHome = e.altKey; // Option key on Mac

    setInsertionIndex(null);

    if (!draggedExpId) return;

    // Check if dropping from different group
    const currentGroupKey = groupId || 'ungrouped';
    if (sourceGroupId !== currentGroupKey) {
      // Moving to different group
      onDragExperiment(draggedExpId, groupId, multiHome);
    } else if (draggedExpId !== targetExpId) {
      // Reordering within same group - use insertion index if available
      // Use String(id) for comparisons since dataTransfer gives strings
      const currentOrder = experiments.map(e => String(e.id));
      const draggedIndex = currentOrder.indexOf(draggedExpId);

      // Use insertion index to determine target position
      let targetIndex = insertionIndex !== null ? insertionIndex : currentOrder.indexOf(targetExpId);

      // Adjust for the removal of the dragged item
      if (draggedIndex < targetIndex) {
        targetIndex--;
      }

      const newOrder = [...currentOrder];
      newOrder.splice(draggedIndex, 1);
      newOrder.splice(targetIndex, 0, draggedExpId);

      onReorderInGroup(groupId, newOrder);
    }
    setDraggedId(null);
  }, [draggedId, experiments, insertionIndex, onDragExperiment, onReorderInGroup, groupId]);

  const handleGroupDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current = 0;
    setIsDragOver(false);
    const draggedExpId = e.dataTransfer.getData('text/plain');
    const sourceGroupId = e.dataTransfer.getData('application/x-source-group');
    const multiHome = e.altKey;

    if (!draggedExpId) return;

    const currentGroupKey = groupId || 'ungrouped';
    if (sourceGroupId !== currentGroupKey) {
      onDragExperiment(draggedExpId, groupId, multiHome);
    }
  }, [groupId, onDragExperiment]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current++;
    if (dragCountRef.current === 1) {
      setIsDragOver(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current--;
    if (dragCountRef.current === 0) {
      setIsDragOver(false);
    }
  }, []);

  const handleSaveGroupName = () => {
    if (group && editedName.trim() && editedName !== group.name) {
      onUpdateGroup(group.id, { name: editedName.trim() });
    }
    setIsEditingName(false);
  };

  // Selected experiments in this group
  const selectedInGroup = useMemo(() => {
    return experiments.filter(e => selectedExperiments.has(String(e.id)));
  }, [experiments, selectedExperiments]);

  const allSelected = selectedInGroup.length === experiments.length && experiments.length > 0;
  const someSelected = selectedInGroup.length > 0 && selectedInGroup.length < experiments.length;

  // Render header rows with hierarchical flag columns
  const renderHeaderRows = () => {
    const totalRows = maxDepth || 1;

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

    if (rowFlags.length > 0) {
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
              onChange={e => onSelectAll(e.target.checked, groupId)}
            />
          </th>
          <th className="col-expand" rowSpan={totalRows}>
            {isUngrouped && (
              <span
                className="create-btn"
                onClick={() => onQuickCreate()}
                title="Create new experiment"
              >
                +
              </span>
            )}
          </th>
          <th className="col-id" rowSpan={totalRows}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Name</span>
              {selectedInGroup.length > 0 && (
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
          {commonNodes === null && <th className="col-resources" rowSpan={totalRows}>Nodes</th>}
          {commonGpus === null && <th className="col-resources" rowSpan={totalRows}>GPUs</th>}
          {rows[0]}
          <th className="col-padding" rowSpan={totalRows}></th>
        </tr>
        {rows.slice(1).map((row, i) => (
          <tr key={i + 1}>{row}</tr>
        ))}
      </>
    );
  };

  return (
    <div
      className={`experiment-group ${isDragOver ? 'drag-over' : ''}`}
      onDragOver={e => e.preventDefault()}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleGroupDrop}
      style={{
        marginBottom: '20px',
        border: isDragOver ? '2px dashed #2196f3' : '2px solid transparent',
        borderRadius: '6px',
        transition: 'border-color 0.15s ease',
      }}
    >
      {/* Group header */}
      <div className="group-header" style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 12px',
        backgroundColor: isDragOver
          ? (isUngrouped ? '#e0e0e0' : '#bbdefb')
          : (isUngrouped ? '#f5f5f5' : '#e3f2fd'),
        borderRadius: '4px 4px 0 0',
        borderBottom: '2px solid #ddd',
        transition: 'background-color 0.15s ease',
      }}>
        {isEditingName ? (
          <input
            type="text"
            value={editedName}
            onChange={e => setEditedName(e.target.value)}
            onBlur={handleSaveGroupName}
            onKeyDown={e => { if (e.key === 'Enter') handleSaveGroupName(); if (e.key === 'Escape') setIsEditingName(false); }}
            autoFocus
            style={{ fontSize: '14px', fontWeight: 600, border: 'none', borderBottom: '1px solid #2196f3', background: 'transparent', padding: '2px 4px' }}
          />
        ) : (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '14px', fontWeight: 600, color: '#333' }}>
              {isUngrouped ? 'Ungrouped' : group?.name}
            </span>
            {!isUngrouped && (
              <span
                onClick={() => { setEditedName(group?.name || ''); setIsEditingName(true); }}
                style={{ cursor: 'pointer', color: '#999', fontSize: '12px' }}
                title="Edit group name"
              >
                ✎
              </span>
            )}
          </span>
        )}
        {/* Common flags display - editable */}
        {(Object.keys(commonFlags).length > 0 || commonNodes !== null || commonGpus !== null || commonToolPath !== null || !isUngrouped) && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginLeft: '8px', alignItems: 'center' }}>
            {/* Tool path first - editable */}
            {commonToolPath !== null && !isUngrouped && (
              isEditingToolPath ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px',
                    backgroundColor: '#fff',
                    border: '1px solid #3f51b5',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                >
                  <input
                    type="text"
                    value={editedToolPath}
                    onChange={e => setEditedToolPath(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        updateToolPathForAll(editedToolPath);
                        setIsEditingToolPath(false);
                        setEditedToolPath('');
                      }
                      if (e.key === 'Escape') { setIsEditingToolPath(false); setEditedToolPath(''); }
                    }}
                    autoFocus
                    placeholder="tool path"
                    style={{
                      border: 'none',
                      outline: 'none',
                      width: '200px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      padding: '2px 4px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '2px',
                    }}
                  />
                  <span
                    onClick={() => {
                      updateToolPathForAll(editedToolPath);
                      setIsEditingToolPath(false);
                      setEditedToolPath('');
                    }}
                    style={{ cursor: 'pointer', color: '#4CAF50', fontWeight: 'bold', padding: '0 2px' }}
                    title="Save"
                  >
                    ✓
                  </span>
                  <span
                    onClick={() => { setIsEditingToolPath(false); setEditedToolPath(''); }}
                    style={{ cursor: 'pointer', color: '#999', padding: '0 2px' }}
                    title="Cancel"
                  >
                    ✕
                  </span>
                </span>
              ) : commonToolPath ? (
                <span
                  onClick={() => { setIsEditingToolPath(true); setEditedToolPath(commonToolPath); }}
                  style={{
                    display: 'inline-block',
                    padding: '6px 10px',
                    backgroundColor: '#e8eaf6',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#3f51b5',
                    whiteSpace: 'nowrap',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#c5cae9')}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = '#e8eaf6')}
                  title="Click to edit tool path"
                >
                  {commonToolPath}
                </span>
              ) : null
            )}
            {/* Regular flags */}
            {Object.entries(commonFlags)
              .sort((a, b) => a[0].localeCompare(b[0]))
              .map(([key, value]) => {
                const shortKey = key.split('.').pop() || key;
                const isEditing = editingFlag === key;

                if (isEditing) {
                  return (
                    <span
                      key={key}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '4px 8px',
                        backgroundColor: '#fff',
                        border: '1px solid #2196f3',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontFamily: 'monospace',
                      }}
                    >
                      <span style={{ color: '#666' }}>{shortKey}:</span>
                      <input
                        type="text"
                        value={editedFlagValue}
                        onChange={e => setEditedFlagValue(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') handleSaveFlagValue(key, editedFlagValue);
                          if (e.key === 'Escape') { setEditingFlag(null); setEditedFlagValue(''); }
                        }}
                        autoFocus
                        style={{
                          border: 'none',
                          outline: 'none',
                          width: '80px',
                          fontSize: '11px',
                          fontFamily: 'monospace',
                          padding: '2px 4px',
                          backgroundColor: '#f5f5f5',
                          borderRadius: '2px',
                        }}
                      />
                      <span
                        onClick={() => handleSaveFlagValue(key, editedFlagValue)}
                        style={{ cursor: 'pointer', color: '#4CAF50', fontWeight: 'bold', padding: '0 2px' }}
                        title="Save"
                      >
                        ✓
                      </span>
                      <span
                        onClick={() => { setEditingFlag(null); setEditedFlagValue(''); }}
                        style={{ cursor: 'pointer', color: '#999', padding: '0 2px' }}
                        title="Cancel"
                      >
                        ✕
                      </span>
                    </span>
                  );
                }

                const isBool = typeof value === 'boolean';
                const isInt = typeof value === 'number' && Number.isInteger(value);
                const displayValue = isBool
                  ? (value ? '✓' : '✗')
                  : isInt
                    ? formatLargeNumber(value as number)
                    : String(value);
                return (
                  <span
                    key={key}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      backgroundColor: '#e0e0e0',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      color: '#555',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                    }}
                  >
                    <span
                      title={`${key} - Click to edit`}
                      onClick={() => {
                        setEditingFlag(key);
                        const initialValue = typeof value === 'boolean'
                          ? String(value)
                          : (typeof value === 'number' && Number.isInteger(value))
                            ? formatLargeNumber(value)
                            : String(value);
                        setEditedFlagValue(initialValue);
                      }}
                      style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '4px', padding: '6px 10px' }}
                    >
                      <span style={{ color: '#777' }}>{shortKey}:</span>
                      <span style={{
                        fontWeight: isBool ? 800 : 600,
                        fontSize: isBool ? '14px' : '11px',
                        color: isBool ? (value ? '#2e7d32' : '#c62828') : undefined,
                        lineHeight: 1,
                      }}>{displayValue}</span>
                    </span>
                    <span
                      onClick={(e) => { e.stopPropagation(); deleteFlagForAll(key); }}
                      style={{
                        cursor: 'pointer',
                        color: '#c62828',
                        fontSize: '10px',
                        padding: '6px 8px',
                        borderLeft: '1px solid #ccc',
                        display: 'inline-flex',
                        alignItems: 'center',
                        transition: 'background-color 0.15s',
                      }}
                      title="Delete flag from all experiments"
                      onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#ffebee')}
                      onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      ✕
                    </span>
                  </span>
                );
              })}
            {/* Nodes pill - after flags */}
            {commonNodes !== null && (
              editingResource === 'nodes' ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px',
                    backgroundColor: '#fff',
                    border: '1px solid #2196f3',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                >
                  <span style={{ color: '#666' }}>nodes:</span>
                  <input
                    type="text"
                    value={editedResourceValue}
                    onChange={e => setEditedResourceValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        const val = parseInt(editedResourceValue);
                        if (!isNaN(val) && val > 0) updateResourceForAll('nodes', val);
                        setEditingResource(null);
                      }
                      if (e.key === 'Escape') { setEditingResource(null); setEditedResourceValue(''); }
                    }}
                    autoFocus
                    style={{
                      border: 'none',
                      outline: 'none',
                      width: '40px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      padding: '2px 4px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '2px',
                    }}
                  />
                  <span
                    onClick={() => {
                      const val = parseInt(editedResourceValue);
                      if (!isNaN(val) && val > 0) updateResourceForAll('nodes', val);
                      setEditingResource(null);
                    }}
                    style={{ cursor: 'pointer', color: '#4CAF50', fontWeight: 'bold', padding: '0 2px' }}
                    title="Save"
                  >
                    ✓
                  </span>
                  <span
                    onClick={() => { setEditingResource(null); setEditedResourceValue(''); }}
                    style={{ cursor: 'pointer', color: '#999', padding: '0 2px' }}
                    title="Cancel"
                  >
                    ✕
                  </span>
                </span>
              ) : (
                <span
                  title="Click to edit nodes"
                  onClick={() => { setEditingResource('nodes'); setEditedResourceValue(String(commonNodes)); }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '6px 10px',
                    backgroundColor: '#e3f2fd',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#555',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                    whiteSpace: 'nowrap',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#bbdefb')}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = '#e3f2fd')}
                >
                  <span style={{ color: '#1565c0' }}>nodes:</span>
                  <span style={{ fontWeight: 600 }}>{commonNodes}</span>
                </span>
              )
            )}
            {/* GPUs pill */}
            {commonGpus !== null && (
              editingResource === 'gpus' ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px',
                    backgroundColor: '#fff',
                    border: '1px solid #2196f3',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                >
                  <span style={{ color: '#666' }}>gpus:</span>
                  <input
                    type="text"
                    value={editedResourceValue}
                    onChange={e => setEditedResourceValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        const val = parseInt(editedResourceValue);
                        if (!isNaN(val) && val > 0) updateResourceForAll('gpus', val);
                        setEditingResource(null);
                      }
                      if (e.key === 'Escape') { setEditingResource(null); setEditedResourceValue(''); }
                    }}
                    autoFocus
                    style={{
                      border: 'none',
                      outline: 'none',
                      width: '40px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      padding: '2px 4px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '2px',
                    }}
                  />
                  <span
                    onClick={() => {
                      const val = parseInt(editedResourceValue);
                      if (!isNaN(val) && val > 0) updateResourceForAll('gpus', val);
                      setEditingResource(null);
                    }}
                    style={{ cursor: 'pointer', color: '#4CAF50', fontWeight: 'bold', padding: '0 2px' }}
                    title="Save"
                  >
                    ✓
                  </span>
                  <span
                    onClick={() => { setEditingResource(null); setEditedResourceValue(''); }}
                    style={{ cursor: 'pointer', color: '#999', padding: '0 2px' }}
                    title="Cancel"
                  >
                    ✕
                  </span>
                </span>
              ) : (
                <span
                  title="Click to edit gpus"
                  onClick={() => { setEditingResource('gpus'); setEditedResourceValue(String(commonGpus)); }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '6px 10px',
                    backgroundColor: '#e3f2fd',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#555',
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                    whiteSpace: 'nowrap',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#bbdefb')}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = '#e3f2fd')}
                >
                  <span style={{ color: '#1565c0' }}>gpus:</span>
                  <span style={{ fontWeight: 600 }}>{commonGpus}</span>
                </span>
              )
            )}
            {/* Add new flag button or input */}
            {!isUngrouped && (
              isAddingFlag ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 8px',
                    backgroundColor: '#fff',
                    border: '1px solid #4CAF50',
                    borderRadius: '4px',
                    fontSize: '11px',
                    position: 'relative',
                  }}
                >
                  <input
                    type="text"
                    value={newFlagKey}
                    onChange={e => {
                      setNewFlagKey(e.target.value);
                      setShowFlagDropdown(true);
                    }}
                    onFocus={() => setShowFlagDropdown(true)}
                    placeholder="Type to search..."
                    autoFocus
                    style={{
                      border: 'none',
                      outline: 'none',
                      width: '120px',
                      fontSize: '11px',
                      padding: '0 2px',
                    }}
                  />
                  {showFlagDropdown && flagDefinitions.length > 0 && (
                    <div
                      onClick={e => e.stopPropagation()}
                      style={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        marginTop: '4px',
                        maxHeight: '200px',
                        overflowY: 'auto',
                        background: 'white',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                        zIndex: 1000,
                        minWidth: '250px',
                      }}
                    >
                      {flagDefinitions
                        .filter(f => !newFlagKey || f.flag.toLowerCase().includes(newFlagKey.toLowerCase()))
                        .slice(0, 10)
                        .map(flagDef => (
                          <div
                            key={flagDef.flag}
                            onClick={() => handleAddFlag(flagDef)}
                            style={{
                              padding: '6px 10px',
                              cursor: 'pointer',
                              borderBottom: '1px solid #f0f0f0',
                              fontSize: '11px',
                            }}
                            onMouseEnter={e => e.currentTarget.style.background = '#f5f5f5'}
                            onMouseLeave={e => e.currentTarget.style.background = 'white'}
                          >
                            <div style={{ fontWeight: 500, fontFamily: 'monospace' }}>{flagDef.flag}</div>
                            <div style={{ fontSize: '10px', color: '#666', marginTop: '2px' }}>
                              <span style={{ color: '#2196f3' }}>{flagDef.type}</span>
                              <span> • <code style={{ background: '#f5f5f5', padding: '1px 3px', borderRadius: '2px' }}>
                                {flagDef.default === null ? 'null' : flagDef.default === undefined ? 'undefined' : String(flagDef.default)}
                              </code></span>
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                  <span style={{ color: '#666' }}>:</span>
                  <input
                    type="text"
                    value={newFlagValue}
                    onChange={e => setNewFlagValue(e.target.value)}
                    placeholder="value"
                    onKeyDown={e => {
                      if (e.key === 'Enter') handleAddFlag();
                      if (e.key === 'Escape') { setIsAddingFlag(false); setNewFlagKey(''); setNewFlagValue(''); setShowFlagDropdown(false); }
                    }}
                    style={{
                      border: 'none',
                      outline: 'none',
                      width: '60px',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                      padding: '0 2px',
                    }}
                  />
                  <span
                    onClick={() => handleAddFlag()}
                    style={{ cursor: 'pointer', color: '#4CAF50', fontWeight: 'bold', fontSize: '14px' }}
                    title="Add flag"
                  >
                    ✓
                  </span>
                  <span
                    onClick={() => { setIsAddingFlag(false); setNewFlagKey(''); setNewFlagValue(''); setShowFlagDropdown(false); }}
                    style={{ cursor: 'pointer', color: '#999', fontSize: '14px' }}
                    title="Cancel"
                  >
                    ✕
                  </span>
                </span>
              ) : (
                <span
                  onClick={() => setIsAddingFlag(true)}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '6px 12px',
                    backgroundColor: '#e8f5e9',
                    borderRadius: '4px',
                    fontSize: '14px',
                    color: '#4CAF50',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                  }}
                  title="Add common flag"
                >
                  +
                </span>
              )
            )}
          </div>
        )}
        {/* Right side of header */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ color: '#666', fontSize: '12px' }}>
            {experiments.length} experiment{experiments.length !== 1 ? 's' : ''}
          </span>
          {experiments.length > 0 && (
            <span
              onClick={() => {
                const commands = experiments.map(exp => buildCommand(exp)).join('\n');
                navigator.clipboard.writeText(commands);
                showNotification(`Copied ${experiments.length} command(s)`, 'success');
              }}
              style={{
                cursor: 'pointer',
                color: '#666',
                fontSize: '14px',
                padding: '2px 4px',
                borderRadius: '3px',
                transition: 'background-color 0.15s',
              }}
              title="Copy all commands"
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = '#e0e0e0')}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
            >
              ⧉
            </span>
          )}
          {isUngrouped ? (
            <span
              onClick={onCreateGroup}
              style={{ cursor: 'pointer', color: '#2196f3', fontSize: '12px' }}
              title="Create new group"
            >
              + New Group
            </span>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span
                onClick={() => onQuickCreate({
                  flags: commonFlags as Record<string, unknown>,
                  nodes: commonNodes ?? undefined,
                  gpus: commonGpus ?? undefined,
                  groupId: group?.id,
                })}
                style={{
                  cursor: 'pointer',
                  color: '#4CAF50',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  padding: '0 4px',
                }}
                title="Add new experiment to group"
              >
                +
              </span>
              {confirmingDelete ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '11px' }}>
                  <span style={{ color: '#c62828' }}>Delete?</span>
                  <span
                    onClick={() => { group && onDeleteGroup(group.id); setConfirmingDelete(false); }}
                    style={{ cursor: 'pointer', color: '#c62828', fontWeight: 'bold' }}
                    title="Confirm delete"
                  >
                    ✓
                  </span>
                  <span
                    onClick={() => setConfirmingDelete(false)}
                    style={{ cursor: 'pointer', color: '#999' }}
                    title="Cancel"
                  >
                    ✕
                  </span>
                </span>
              ) : (
                <span
                  onClick={() => setConfirmingDelete(true)}
                  style={{
                    cursor: 'pointer',
                    color: '#999',
                    fontSize: '14px',
                    padding: '0 4px',
                  }}
                  title="Delete group"
                >
                  ✕
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Experiments table */}
      <div className="table-wrapper" style={{ border: '1px solid #ddd', borderTop: 'none', borderRadius: '0 0 4px 4px' }}>
        <table id={`experiments-table-${groupId || 'ungrouped'}`}>
          <thead>{renderHeaderRows()}</thead>
          <tbody ref={tbodyRef}>
            {experiments.length === 0 ? (
              <tr>
                <td colSpan={100} className="empty-state" style={{ padding: '20px', textAlign: 'center', color: '#999' }}>
                  {isUngrouped ? 'No experiments. Click "+" to create one.' : 'Drag experiments here to add them to this group.'}
                </td>
              </tr>
            ) : (
              experiments.map((exp, index) => {
                const expIdStr = String(exp.id);
                const isExpanded = expandedExperiments.has(expIdStr);
                const isSelected = selectedExperiments.has(expIdStr);
                const isDragging = draggedId === expIdStr;
                const showInsertBefore = insertionIndex === index && draggedId !== null;
                const showInsertAfter = insertionIndex === index + 1 && index === experiments.length - 1 && draggedId !== null;

                return [
                  // Insertion line before row
                  showInsertBefore && (
                    <tr key={`insert-${exp.id}`} className="insertion-line">
                      <td colSpan={100} style={{ padding: 0, border: 'none' }}>
                        <div style={{ height: '3px', backgroundColor: '#2196f3', margin: '0' }} />
                      </td>
                    </tr>
                  ),
                  <tr
                    key={exp.id}
                    className={`main-row ${isDragging ? 'dragging' : ''}`}
                    data-exp-id={exp.id}
                    draggable
                    onDragStart={(e) => handleDragStart(String(exp.id), e)}
                    onDragEnd={handleDragEnd}
                    onDragOver={(e) => handleRowDragOver(String(exp.id), e)}
                    onDrop={(e) => handleDrop(String(exp.id), e)}
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
                        onClick={handleCopy(`https://wandb.ai/metta-research/metta/runs/${exp.name}`, 'W&B URL', `https://wandb.ai/metta-research/metta/runs/${exp.name}`)}
                        title="Click to copy, Cmd+click to open"
                      >
                        W&B
                      </span>
                      <span
                        className="wandb-link"
                        onClick={handleCopy(`https://app.datadoghq.com/logs?query=metta_run_id%3A%22${exp.name}%22`, 'log URL', `https://app.datadoghq.com/logs?query=metta_run_id%3A%22${exp.name}%22`)}
                        title="Click to copy, Cmd+click to open"
                      >
                        log
                      </span>
                      <span
                        onClick={handleCopy(exp.name, 'name')}
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
                    {commonNodes === null && <td className="col-resources">{exp.nodes}</td>}
                    {commonGpus === null && <td className="col-resources">{exp.gpus}</td>}
                    {flagColumns.map(flag => (
                      <td key={flag} className="col-flag">
                        {formatFlagValue(exp.flags[flag])}
                      </td>
                    ))}
                    <td className="col-padding"></td>
                  </tr>,
                  <tr
                    key={`${exp.id}-expanded`}
                    className={`expanded-row ${isExpanded ? 'show' : ''}`}
                    data-exp-id={exp.id}
                  >
                    <td colSpan={7 + flagColumns.length + (commonNodes === null ? 1 : 0) + (commonGpus === null ? 1 : 0)}>
                      {isExpanded && renderExpandedDetails(exp)}
                    </td>
                  </tr>,
                  // Insertion line after last row
                  showInsertAfter && (
                    <tr key={`insert-after-${exp.id}`} className="insertion-line">
                      <td colSpan={100} style={{ padding: 0, border: 'none' }}>
                        <div style={{ height: '3px', backgroundColor: '#2196f3', margin: '0' }} />
                      </td>
                    </tr>
                  ),
                ];
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
