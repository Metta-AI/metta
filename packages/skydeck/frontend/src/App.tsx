import { useState, useEffect, useCallback, useRef } from 'react';
import { NotificationProvider } from './hooks/useNotifications';
import { useNotifications } from './hooks/useNotifications';
import { useApi } from './hooks/useApi';
import type { Experiment, ExperimentGroup, Job, HealthData } from './types';
import { HealthStatus } from './components/HealthStatus';
import { ExperimentGroupView } from './components/ExperimentGroupView';
import { ExpandedDetails } from './components/ExpandedDetails';
import { JobsTable } from './components/JobsTable';
import { Notifications } from './components/Notifications';
import './App.css';

const MY_USER_ID = 'daveey';

function AppContent() {
  const { apiCall } = useApi();
  const { showNotification } = useNotifications();

  // Data state
  const [groups, setGroups] = useState<ExperimentGroup[]>([]);
  const [ungroupedExperiments, setUngroupedExperiments] = useState<Experiment[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [health, setHealth] = useState<HealthData | null>(null);

  // UI state
  const [expandedExperiments, setExpandedExperiments] = useState<Set<string>>(new Set());
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string>>(new Set());
  const [editingConfigs, setEditingConfigs] = useState<Set<string>>(new Set());

  // Jobs filters
  const [showStoppedJobs, setShowStoppedJobs] = useState(false);
  const [showOrphanedOnly, setShowOrphanedOnly] = useState(false);
  const [showMyJobsOnly, setShowMyJobsOnly] = useState(true);
  const [jobsLimit, setJobsLimit] = useState(20);
  const [jobsFilterText, setJobsFilterText] = useState('');

  // Timing
  const lastSyncRef = useRef(Date.now());
  const [backendStaleness, setBackendStaleness] = useState(0);

  // Get all experiments (from groups + ungrouped)
  const allExperiments = useCallback(() => {
    const fromGroups = groups.flatMap(g => g.experiments);
    return [...fromGroups, ...ungroupedExperiments];
  }, [groups, ungroupedExperiments]);

  // Load data
  const loadData = useCallback(async () => {
    // Skip refresh if any config panels are in edit mode
    if (editingConfigs.size > 0) {
      return;
    }

    try {
      const [healthData, groupsData, jobsData] = await Promise.all([
        apiCall<HealthData>('/health'),
        apiCall<{ groups: ExperimentGroup[]; ungrouped: Experiment[] }>('/groups'),
        apiCall<{ jobs: Job[] }>(`/skypilot-jobs?limit=${jobsLimit}&include_stopped=${showStoppedJobs}`),
      ]);

      setHealth(healthData);
      setGroups(groupsData.groups || []);
      setUngroupedExperiments(groupsData.ungrouped || []);
      setJobs(jobsData.jobs || []);

      // Sync expanded state from backend (use String(id) for consistency)
      const newExpanded = new Set<string>();
      const allExps = [...(groupsData.groups || []).flatMap(g => g.experiments), ...(groupsData.ungrouped || [])];
      allExps.forEach((exp: Experiment) => {
        if (exp.is_expanded) {
          newExpanded.add(String(exp.id));
        }
      });
      setExpandedExperiments(newExpanded);

      lastSyncRef.current = Date.now();
    } catch (error) {
      console.error('Error loading data:', error);
    }
  }, [apiCall, jobsLimit, showStoppedJobs, editingConfigs.size]);

  // Load jobs settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await fetch('/api/settings/jobs-filters');
        if (response.ok) {
          const data = await response.json();
          const settings = data.value || {};
          setShowStoppedJobs(settings.showStopped || false);
          setShowOrphanedOnly(settings.showOrphaned || false);
          setJobsLimit(settings.limit || 20);
        }
      } catch {
        // Use defaults
      }
    };
    loadSettings();
  }, []);

  // Initial load and polling
  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Update backend staleness every second
  useEffect(() => {
    const interval = setInterval(() => {
      setBackendStaleness((Date.now() - lastSyncRef.current) / 1000);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Save jobs setting helper
  const saveJobsSetting = async (key: string, value: unknown) => {
    try {
      let settings: Record<string, unknown> = {};
      try {
        const response = await fetch('/api/settings/jobs-filters');
        if (response.ok) {
          const data = await response.json();
          settings = data.value || {};
        }
      } catch {
        // No existing settings
      }
      settings[key] = value;
      await fetch('/api/settings/jobs-filters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
    } catch (error) {
      console.error('Error saving jobs setting:', error);
    }
  };

  // Experiment actions
  const setDesiredState = async (experimentId: string, state: 'RUNNING' | 'STOPPED') => {
    try {
      await apiCall(`/experiments/${experimentId}/state`, {
        method: 'POST',
        body: JSON.stringify({ desired_state: state }),
      });
      await loadData();
    } catch (error) {
      console.error('Error updating state:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error updating state: ${msg}`, 'error');
    }
  };

  const toggleExpanded = async (experimentId: string) => {
    const isExpanding = !expandedExperiments.has(experimentId);
    setExpandedExperiments(prev => {
      const next = new Set(prev);
      if (isExpanding) {
        next.add(experimentId);
      } else {
        next.delete(experimentId);
      }
      return next;
    });

    // Persist to backend
    try {
      await apiCall(`/experiments/${experimentId}/expanded`, {
        method: 'POST',
        body: JSON.stringify({ is_expanded: isExpanding }),
      });
    } catch (error) {
      console.error('Error saving expanded state:', error);
    }
  };

  const toggleStar = async (experimentId: string) => {
    const exp = allExperiments().find(e => String(e.id) === experimentId);
    if (!exp) return;

    try {
      await apiCall(`/experiments/${experimentId}/starred`, {
        method: 'POST',
        body: JSON.stringify({ starred: !exp.starred }),
      });
      // Optimistically update local state
      setGroups(prev => prev.map(g => ({
        ...g,
        experiments: g.experiments.map(e => String(e.id) === experimentId ? { ...e, starred: !e.starred } : e)
      })));
      setUngroupedExperiments(prev => prev.map(e => String(e.id) === experimentId ? { ...e, starred: !e.starred } : e));
    } catch (error) {
      console.error('Error toggling star:', error);
      showNotification('Failed to toggle star', 'error');
    }
  };

  const deleteExperiment = async (experimentId: string) => {
    try {
      await apiCall(`/experiments/${experimentId}`, { method: 'DELETE' });

      // Close any editing panels for this experiment
      setEditingConfigs(prev => {
        const next = new Set(prev);
        next.delete(experimentId);
        return next;
      });

      showNotification(`Deleted experiment ${experimentId}`, 'success', async () => {
        try {
          await apiCall(`/experiments/${experimentId}/undelete`, { method: 'POST' });
          showNotification(`Restored experiment ${experimentId}`, 'success');
          await loadData();
        } catch {
          showNotification('Failed to restore experiment', 'error');
        }
      });
      await loadData();
    } catch (error) {
      console.error('Error deleting experiment:', error);
      showNotification('Failed to delete experiment', 'error');
    }
  };

  const quickCreateExperiment = async (options?: { flags?: Record<string, unknown>; nodes?: number; gpus?: number; groupId?: string }) => {
    try {
      const response = await apiCall<{ experiment: { id: string; name: string } }>('/experiments', {
        method: 'POST',
        body: JSON.stringify({
          name: 'New Experiment',
          base_command: 'lt',
          tool_path: 'recipes.experiment.cog_arena.train',
          nodes: options?.nodes ?? 1,
          gpus: options?.gpus ?? 0,
          flags: options?.flags ?? {},
          desired_state: 'STOPPED',
        }),
      });

      // If groupId provided, add to that group
      if (options?.groupId) {
        await apiCall(`/groups/${options.groupId}/experiments`, {
          method: 'POST',
          body: JSON.stringify({ experiment_ids: [response.experiment.id], multi_home: false }),
        });
      }

      showNotification(`Created experiment: ${response.experiment.name}`, 'success');
      await loadData();
    } catch (error) {
      console.error('Error creating experiment:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error creating experiment: ${msg}`, 'error');
    }
  };

  const duplicateExperiment = async (experimentId: string, skipReload = false) => {
    const exp = allExperiments().find(e => String(e.id) === experimentId);
    if (!exp) {
      throw new Error(`Experiment ${experimentId} not found`);
    }

    try {
      // Create the duplicate experiment
      const response = await apiCall<{ experiment: { id: string; name: string } }>('/experiments', {
        method: 'POST',
        body: JSON.stringify({
          name: `${exp.name} (copy)`,
          base_command: exp.base_command,
          tool_path: exp.tool_path,
          git_branch: exp.git_branch,
          nodes: exp.nodes,
          gpus: exp.gpus,
          instance_type: exp.instance_type,
          cloud: exp.cloud,
          spot: exp.spot,
          flags: exp.flags,
          description: exp.description,
          tags: exp.tags,
          group: null,  // Don't set group in creation - we'll add it manually below
          desired_state: 'STOPPED',
          order: (exp.exp_order ?? 0) + 1,  // Place it right after the original
        }),
      });

      const newExpId = String(response.experiment.id);

      // Find all groups containing the original experiment
      const groupsContainingExp = groups.filter(g =>
        g.experiments.some(e => String(e.id) === experimentId)
      );

      // Check if it's in the ungrouped list
      const isUngrouped = ungroupedExperiments.some(e => String(e.id) === experimentId);

      if (groupsContainingExp.length > 0) {
        // Add to each group and reorder to place right after original
        for (const group of groupsContainingExp) {
          // Add to group first
          await apiCall(`/groups/${group.id}/experiments`, {
            method: 'POST',
            body: JSON.stringify({
              experiment_ids: [newExpId],
              multi_home: false,
            }),
          });

          // Now reorder - insert the duplicate right after the original
          const expIndex = group.experiments.findIndex(e => String(e.id) === experimentId);
          if (expIndex !== -1) {
            const newOrder = [
              ...group.experiments.slice(0, expIndex + 1).map(e => String(e.id)),
              newExpId,
              ...group.experiments.slice(expIndex + 1).map(e => String(e.id))
            ];

            await apiCall(`/groups/${group.id}/reorder`, {
              method: 'POST',
              body: JSON.stringify({ order: newOrder }),
            });
          }
        }
      } else if (isUngrouped) {
        // Reorder ungrouped experiments to place duplicate right after original
        const expIndex = ungroupedExperiments.findIndex(e => String(e.id) === experimentId);
        if (expIndex !== -1) {
          const newOrder = [
            ...ungroupedExperiments.slice(0, expIndex + 1).map(e => String(e.id)),
            newExpId,
            ...ungroupedExperiments.slice(expIndex + 1).map(e => String(e.id))
          ];

          await apiCall('/experiments/reorder', {
            method: 'POST',
            body: JSON.stringify({ order: newOrder }),
          });
        }
      }

      if (!skipReload) {
        showNotification(`Created duplicate: ${response.experiment.name}`, 'success');
        await loadData();
      }

      return response.experiment.name;
    } catch (error) {
      console.error('Error duplicating experiment:', error);
      const msg = error instanceof Error ? error.message : String(error);
      if (!skipReload) {
        showNotification(`Error duplicating experiment: ${msg}`, 'error');
      }
      throw error;
    }
  };

  // Group management
  const createGroup = async () => {
    try {
      const response = await apiCall<{ group: ExperimentGroup }>('/groups', {
        method: 'POST',
        body: JSON.stringify({ name: 'New Group', flags: [] }),
      });
      showNotification(`Created group: ${response.group.name}`, 'success');
      await loadData();
    } catch (error) {
      console.error('Error creating group:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error creating group: ${msg}`, 'error');
    }
  };

  const updateGroup = async (groupId: string, updates: { name?: string; collapsed?: boolean }) => {
    try {
      await apiCall(`/groups/${groupId}`, {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
      // Optimistically update local state
      setGroups(prev => prev.map(g => g.id === groupId ? { ...g, ...updates } : g));
    } catch (error) {
      console.error('Error updating group:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error updating group: ${msg}`, 'error');
    }
  };

  const deleteGroup = async (groupId: string) => {
    try {
      await apiCall(`/groups/${groupId}`, { method: 'DELETE' });
      showNotification('Group deleted', 'success');
      await loadData();
    } catch (error) {
      console.error('Error deleting group:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error deleting group: ${msg}`, 'error');
    }
  };

  const dragExperimentToGroup = async (experimentId: string, targetGroupId: string | null, multiHome: boolean) => {
    try {
      if (targetGroupId === null) {
        // Moving to ungrouped - need to remove from all groups
        const groupsForExp = groups.filter(g => g.experiments.some(e => String(e.id) === experimentId));
        for (const group of groupsForExp) {
          await apiCall(`/groups/${group.id}/experiments/${experimentId}`, { method: 'DELETE' });
        }
      } else {
        // Moving to a group
        await apiCall(`/groups/${targetGroupId}/experiments`, {
          method: 'POST',
          body: JSON.stringify({ experiment_ids: [experimentId], multi_home: multiHome }),
        });
      }
      await loadData();
    } catch (error) {
      console.error('Error moving experiment to group:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error moving experiment: ${msg}`, 'error');
    }
  };

  const reorderInGroup = async (groupId: string | null, order: string[]) => {
    try {
      if (groupId === null) {
        // Reorder ungrouped experiments
        await fetch('/api/experiments/reorder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ order }),
        });
      } else {
        await fetch(`/api/groups/${groupId}/reorder`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ order }),
        });
      }
    } catch (error) {
      console.error('Error reordering:', error);
    }
  };

  // Bulk actions
  const bulkStart = async () => {
    try {
      await Promise.all(
        Array.from(selectedExperiments).map(id =>
          apiCall(`/experiments/${id}/state`, {
            method: 'POST',
            body: JSON.stringify({ desired_state: 'RUNNING' }),
          })
        )
      );
      showNotification(`Started ${selectedExperiments.size} experiments`, 'success');
      await loadData();
    } catch (error) {
      console.error('Error starting experiments:', error);
    }
  };

  const bulkStop = async () => {
    try {
      await Promise.all(
        Array.from(selectedExperiments).map(id =>
          apiCall(`/experiments/${id}/state`, {
            method: 'POST',
            body: JSON.stringify({ desired_state: 'STOPPED' }),
          })
        )
      );
      showNotification(`Stopped ${selectedExperiments.size} experiments`, 'success');
      await loadData();
    } catch (error) {
      console.error('Error stopping experiments:', error);
    }
  };

  const bulkDuplicate = async () => {
    const count = selectedExperiments.size;
    try {
      const duplicatedNames: string[] = [];
      for (const experimentId of selectedExperiments) {
        const name = await duplicateExperiment(experimentId, true);
        duplicatedNames.push(name);
      }
      await loadData();
      showNotification(`Duplicated ${count} experiment(s)`, 'success');
    } catch (error) {
      console.error('Error during bulk duplicate:', error);
      const msg = error instanceof Error ? error.message : String(error);
      showNotification(`Error duplicating experiments: ${msg}`, 'error');
      await loadData(); // Reload anyway to show partial progress
    }
  };

  const bulkDelete = async () => {
    const deletedIds = Array.from(selectedExperiments);
    try {
      await Promise.all(deletedIds.map(id => apiCall(`/experiments/${id}`, { method: 'DELETE' })));
      setSelectedExperiments(new Set());

      showNotification(`Deleted ${deletedIds.length} experiment(s)`, 'success', async () => {
        try {
          await Promise.all(
            deletedIds.map(id => apiCall(`/experiments/${id}/undelete`, { method: 'POST' }))
          );
          showNotification(`Restored ${deletedIds.length} experiment(s)`, 'success');
          await loadData();
        } catch {
          showNotification('Failed to restore experiments', 'error');
        }
      });

      await loadData();
    } catch (error) {
      console.error('Error deleting experiments:', error);
    }
  };

  // Filter jobs
  const filteredJobs = jobs.filter(job => {
    if (showOrphanedOnly) {
      // Jobs match experiments by name, not id
      const experimentNames = new Set(allExperiments().map(e => e.name));
      if (experimentNames.has(job.experiment_id)) return false;
    }
    if (showMyJobsOnly && !job.experiment_id.startsWith(MY_USER_ID + '.')) {
      return false;
    }
    if (jobsFilterText) {
      const searchLower = jobsFilterText.toLowerCase();
      if (
        !job.experiment_id.toLowerCase().includes(searchLower) &&
        !job.id.toString().includes(searchLower)
      ) {
        return false;
      }
    }
    return true;
  });

  // Common props for ExperimentGroupView
  const groupViewProps = {
    expandedExperiments,
    selectedExperiments,
    allGroups: groups,
    onToggleExpanded: toggleExpanded,
    onToggleStar: toggleStar,
    onSetDesiredState: setDesiredState,
    onDelete: deleteExperiment,
    onDuplicate: duplicateExperiment,
    onQuickCreate: quickCreateExperiment,
    onSelectExperiment: (id: string, selected: boolean) => {
      setSelectedExperiments(prev => {
        const next = new Set(prev);
        if (selected) {
          next.add(id);
        } else {
          next.delete(id);
        }
        return next;
      });
    },
    onSelectAll: (selectAll: boolean, groupId: string | null) => {
      const experiments = groupId
        ? groups.find(g => g.id === groupId)?.experiments || []
        : ungroupedExperiments;
      if (selectAll) {
        setSelectedExperiments(prev => {
          const next = new Set(prev);
          experiments.forEach(e => next.add(String(e.id)));
          return next;
        });
      } else {
        setSelectedExperiments(prev => {
          const next = new Set(prev);
          experiments.forEach(e => next.delete(String(e.id)));
          return next;
        });
      }
    },
    onBulkStart: bulkStart,
    onBulkStop: bulkStop,
    onBulkDuplicate: bulkDuplicate,
    onBulkDelete: bulkDelete,
    onSetEditingConfig: (id: string, editing: boolean) => {
      setEditingConfigs(prev => {
        const next = new Set(prev);
        if (editing) {
          next.add(id);
        } else {
          next.delete(id);
        }
        return next;
      });
    },
    onRefreshData: loadData,
    onDragExperiment: dragExperimentToGroup,
    onReorderInGroup: reorderInGroup,
    onUpdateGroup: updateGroup,
    onDeleteGroup: deleteGroup,
    onCreateGroup: createGroup,
    renderExpandedDetails: (experiment: Experiment) => (
      <ExpandedDetails
        experiment={experiment}
        onSetEditingConfig={(id: string, editing: boolean) => {
          setEditingConfigs(prev => {
            const next = new Set(prev);
            if (editing) {
              next.add(id);
            } else {
              next.delete(id);
            }
            return next;
          });
        }}
        onRefreshData={loadData}
      />
    ),
  };

  return (
    <div className="container">
      <header>
        <div className="header-right">
          <HealthStatus health={health} backendStaleness={backendStaleness} />
        </div>
      </header>

      <section id="experiments-section">
        {/* Render each group */}
        {groups.map(group => (
          <ExperimentGroupView
            key={group.id}
            group={group}
            experiments={group.experiments}
            {...groupViewProps}
          />
        ))}

        {/* Render ungrouped experiments */}
        <ExperimentGroupView
          key="ungrouped"
          group={null}
          experiments={ungroupedExperiments}
          {...groupViewProps}
        />
      </section>

      <section id="jobs-section">
        <JobsTable
          jobs={filteredJobs}
          showStoppedJobs={showStoppedJobs}
          showOrphanedOnly={showOrphanedOnly}
          showMyJobsOnly={showMyJobsOnly}
          jobsLimit={jobsLimit}
          jobsFilterText={jobsFilterText}
          onToggleStoppedFilter={async checked => {
            setShowStoppedJobs(checked);
            await saveJobsSetting('showStopped', checked);
          }}
          onToggleOrphanedFilter={async checked => {
            setShowOrphanedOnly(checked);
            await saveJobsSetting('showOrphaned', checked);
          }}
          onToggleMyJobsFilter={async checked => {
            setShowMyJobsOnly(checked);
            await saveJobsSetting('showMyJobs', checked);
          }}
          onUpdateJobsLimit={async value => {
            setJobsLimit(value);
            await saveJobsSetting('limit', value);
          }}
          onFilterJobs={setJobsFilterText}
        />
      </section>

      <Notifications />
    </div>
  );
}

function App() {
  return (
    <NotificationProvider>
      <AppContent />
    </NotificationProvider>
  );
}

export default App;
