// Global state
let currentExperimentId = null;
let allExperiments = [];
let allJobs = [];
let expandedRows = new Set();
let expandedJobs = new Set();
let editingRows = new Set();  // Track which experiments have config panel open in edit mode
let stateEditingRows = new Set();  // Track which experiments have state cell in edit mode
let selectedExperiments = new Set();
let selectedJobs = new Set();
let lastSyncTime = Date.now();
let draggedRow = null;
let jobsPanels = new Map(); // Map of experimentId -> ScrollablePanel for jobs panels
let lastHealthData = null; // Cache health data for continuous staleness updates

// Jobs filters and settings
let showStoppedJobs = false;
let showOrphanedOnly = false;
let showMyJobsOnly = true; // Default to showing only user's jobs
let jobsLimit = 20;
let jobsFilterText = '';
let lastJobsHash = ''; // Cache hash to prevent unnecessary table rebuilds
const MY_USER_ID = 'daveey'; // User ID prefix for filtering

// ==============================================================================
// ScrollablePanel - Reusable panel abstraction with automatic scroll preservation
// ==============================================================================

class ScrollablePanel {
    constructor(elementId) {
        this.elementId = elementId;
        this.scrollPosition = 0;
    }

    /**
     * Get the DOM element for this panel
     */
    getElement() {
        return document.getElementById(this.elementId);
    }

    /**
     * Save current scroll position
     */
    saveScrollPosition() {
        const element = this.getElement();
        if (element) {
            this.scrollPosition = element.scrollTop;
        }
    }

    /**
     * Restore scroll position
     */
    restoreScrollPosition() {
        const element = this.getElement();
        if (element) {
            requestAnimationFrame(() => {
                element.scrollTop = this.scrollPosition;
            });
        }
    }

    /**
     * Update panel content while preserving scroll position
     * @param {string|Function} content - HTML string or function that returns HTML
     */
    update(content) {
        const element = this.getElement();
        if (!element) return;

        // Save scroll position before update
        this.saveScrollPosition();

        // Update content
        if (typeof content === 'function') {
            element.innerHTML = content();
        } else {
            element.innerHTML = content;
        }

        // Restore scroll position after update
        this.restoreScrollPosition();
    }

    /**
     * Create a scrollable panel container HTML
     * @param {string} id - Element ID
     * @param {string} initialContent - Initial HTML content
     * @param {string} maxHeight - Max height CSS value (default: '300px')
     * @returns {string} HTML string for the panel container
     */
    static createContainer(id, initialContent = '', maxHeight = '300px') {
        return `<div id="${id}" class="scrollable-panel" style="max-height: ${maxHeight}; overflow-y: auto;">${initialContent}</div>`;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadJobsSettings();
    loadData();
    setInterval(loadData, 5000); // Refresh every 5 seconds

    // Update staleness counters continuously (every second)
    setInterval(() => {
        if (lastHealthData) {
            updateHealth(lastHealthData);
        }
    }, 1000);
});

// Notification system
function showNotification(message, type = 'info', undoCallback = null) {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;

    // Create message span
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;
    notification.appendChild(messageSpan);

    // Add undo button if callback provided
    if (undoCallback) {
        const undoBtn = document.createElement('button');
        undoBtn.textContent = 'Undo';
        undoBtn.style.marginLeft = '10px';
        undoBtn.style.padding = '2px 8px';
        undoBtn.style.cursor = 'pointer';
        undoBtn.onclick = () => {
            undoCallback();
            notification.remove();
        };
        notification.appendChild(undoBtn);
    }

    container.appendChild(notification);

    // Auto-dismiss after 5 seconds
    const timeout = setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, 5000);

    // Clear timeout if undo is clicked
    if (undoCallback) {
        notification.querySelector('button').addEventListener('click', () => clearTimeout(timeout));
    }
}

// Copy to clipboard utility
function copyToClipboard(text, event) {
    if (event) {
        event.stopPropagation();
        event.preventDefault();
    }

    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showNotification('Failed to copy to clipboard', 'error');
    });
}

// Copy URL on click, open in new tab on command/ctrl-click
function handleLinkPill(url, event) {
    if (event) {
        event.stopPropagation();
        event.preventDefault();
    }

    // Check if command (Mac) or ctrl (Windows/Linux) key is pressed
    if (event && (event.metaKey || event.ctrlKey)) {
        window.open(url, '_blank');
    } else {
        copyToClipboard(url, event);
    }
}

// API calls
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`/api${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API call failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
        throw error;
    }
}

async function loadData() {
    try {
        // Save scroll position before updating
        const scrollY = window.scrollY || window.pageYOffset;

        // Don't refresh if any config panels are in edit mode to prevent disruption
        if (editingRows.size > 0) {
            console.log('Skipping refresh while config panels are being edited');
            return;
        }

        const [health, experiments, skypilotData] = await Promise.all([
            apiCall('/health'),
            apiCall('/experiments'),
            apiCall(`/skypilot-jobs?limit=${jobsLimit}&include_stopped=${showStoppedJobs}`),
        ]);

        // Cache health data for continuous staleness updates
        lastHealthData = health;
        updateHealth(health);
        allExperiments = experiments.experiments || [];
        allJobs = skypilotData.jobs || [];

        // Update last sync time for backend staleness tracking
        lastSyncTime = Date.now();

        // Load expanded state from database
        allExperiments.forEach(exp => {
            if (exp.is_expanded) {
                expandedRows.add(exp.id);
            } else {
                expandedRows.delete(exp.id);
            }
        });

        updateExperimentsTable(allExperiments);
        updateJobsTable(allJobs);

        lastSyncTime = Date.now();

        // Restore scroll position after updating
        window.scrollTo(0, scrollY);
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function updateHealth(health) {
    const indicator = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');

    if (health.status === 'ok') {
        indicator.style.display = 'none'; // Hide the dot indicator

        // Format staleness for each backend as a pill
        const formatBackend = (name, backend, intervalSeconds) => {
            if (!backend || backend.staleness_seconds === null || backend.staleness_seconds === undefined) {
                return `<span style="display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; background-color: #f5f5f5; color: #999; margin-right: 6px;">${name}: —</span>`;
            }

            const staleness = backend.staleness_seconds;
            const threshold1 = intervalSeconds * 1.2;
            const threshold2 = intervalSeconds * 3.0;

            let bgColor, textColor;
            if (staleness < threshold1) {
                bgColor = '#E8F5E9'; // light green
                textColor = '#2E7D32'; // dark green
            } else if (staleness < threshold2) {
                bgColor = '#FFF3E0'; // light orange
                textColor = '#E65100'; // dark orange
            } else {
                bgColor = '#FFEBEE'; // light red
                textColor = '#C62828'; // dark red
            }

            return `<span style="display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; background-color: ${bgColor}; color: ${textColor}; margin-right: 6px;">${name}: ${staleness.toFixed(1)}s</span>`;
        };

        // Calculate backend (web->backend) staleness
        const backendStaleness = (Date.now() - lastSyncTime) / 1000;
        const backendObj = { staleness_seconds: backendStaleness };

        const backend = formatBackend('backend', backendObj, 5); // 5s refresh interval
        const skypilot = formatBackend('skypilot', health.skypilot, 30);
        const s3 = formatBackend('s3', health.s3, 60);
        const obs = formatBackend('obs', health.observatory, 60);

        text.innerHTML = `${backend}${skypilot}${s3}${obs}<span style="margin-left: 8px;">${health.num_experiments} experiments | ${health.num_running_jobs} active jobs</span>`;
    } else {
        indicator.style.display = 'inline-block';
        indicator.classList.add('error');
        text.innerHTML = 'Error';
    }
}

function updateExperimentsTable(experiments) {
    if (!experiments || experiments.length === 0) {
        document.getElementById('experiments-tbody').innerHTML =
            '<tr><td colspan="100" class="empty-state">No experiments found. Click the "+" button to create one.</td></tr>';
        return;
    }

    // Collect all unique flags across all experiments
    const allFlags = new Set();
    experiments.forEach(exp => {
        Object.keys(exp.flags || {}).forEach(flag => allFlags.add(flag));
    });
    const flagColumns = Array.from(allFlags).sort();

    // Backend already provides experiments sorted by exp_order ASC
    // No need to re-sort on the frontend

    // Update table header and get the actual column order from the tree
    const actualColumnOrder = updateTableHeader(flagColumns);

    // Restore bulk actions visibility after header rebuild
    updateBulkActions();

    // Update table body
    const tbody = document.getElementById('experiments-tbody');

    // For rows with config panel in edit mode, keep them completely untouched
    const preservedExpIds = new Set(editingRows);

    // Build a map of existing rows
    const existingRows = new Map();
    const allRows = Array.from(tbody.querySelectorAll('tr'));
    allRows.forEach(row => {
        const expId = row.dataset.expId;
        if (expId) {
            if (!existingRows.has(expId)) {
                existingRows.set(expId, []);
            }
            existingRows.get(expId).push(row);
        }
    });

    // Remove rows that don't have config panel in edit mode
    allRows.forEach(row => {
        const expId = row.dataset.expId;
        if (!expId || !editingRows.has(expId)) {
            row.remove();
        }
    });

    // Add/update rows in correct order
    let lastInsertedNode = null;
    experiments.forEach(exp => {
        if (editingRows.has(exp.id)) {
            // Config panel is in edit mode - skip it but track position
            if (existingRows.has(exp.id)) {
                lastInsertedNode = existingRows.get(exp.id)[existingRows.get(exp.id).length - 1];
            }
        } else {
            // Create new rows
            const mainRow = createMainRow(exp, actualColumnOrder);
            const expandedRow = createExpandedRow(exp, actualColumnOrder.length);

            // Insert in the correct position
            if (lastInsertedNode && lastInsertedNode.parentNode === tbody) {
                lastInsertedNode.after(mainRow);
                mainRow.after(expandedRow);
            } else {
                tbody.appendChild(mainRow);
                tbody.appendChild(expandedRow);
            }

            lastInsertedNode = expandedRow;

            // Always load job history and checkpoints for all rows (they'll be hidden if not expanded)
            loadJobHistory(exp.id);
            loadCheckpoints(exp.id);

            // Restore state edit mode if this experiment was being edited
            if (stateEditingRows.has(exp.id)) {
                const viewSpan = document.getElementById(`state-view-${exp.id}`);
                const editSpan = document.getElementById(`state-edit-${exp.id}`);
                if (viewSpan && editSpan) {
                    viewSpan.style.display = 'none';
                    editSpan.style.display = 'inline-flex';
                }
            }
        }
    });
}

function buildFlagHierarchy(flagColumns) {
    // Build a tree structure from flag paths
    const tree = {};

    flagColumns.forEach(flag => {
        const parts = flag.split('.');
        let current = tree;

        parts.forEach((part, idx) => {
            if (!current[part]) {
                current[part] = { _flags: [], _children: {} };
            }
            if (idx === parts.length - 1) {
                current[part]._flags.push(flag);
            } else {
                current = current[part]._children;
            }
        });
    });

    return tree;
}

function findCommonPrefix(flagColumns) {
    if (flagColumns.length === 0) return '';

    // Find longest common prefix
    let prefix = flagColumns[0];
    for (let i = 1; i < flagColumns.length; i++) {
        while (!flagColumns[i].startsWith(prefix)) {
            prefix = prefix.substring(0, prefix.lastIndexOf('.'));
            if (prefix === '') break;
        }
    }

    // Only use prefix if it ends at a dot
    if (prefix && flagColumns[0][prefix.length] === '.') {
        return prefix + '.';
    }
    return prefix.includes('.') ? prefix.substring(0, prefix.lastIndexOf('.') + 1) : '';
}

function groupFlagsByPrefix(flagColumns) {
    // Build a hierarchical grouping structure
    // Group by common prefixes, creating subgroups when multiple flags share deeper prefixes

    const tree = {};
    const tempByFirstLevel = {};

    // First pass: categorize by first level to see if we should group 2-level flags
    flagColumns.forEach(flag => {
        const parts = flag.split('.');
        const firstLevel = parts[0];
        if (!tempByFirstLevel[firstLevel]) {
            tempByFirstLevel[firstLevel] = [];
        }
        tempByFirstLevel[firstLevel].push(flag);
    });

    // Second pass: build the tree
    flagColumns.forEach(flag => {
        const parts = flag.split('.');

        if (parts.length === 1) {
            // Single level - ungrouped
            if (!tree['_ungrouped']) {
                tree['_ungrouped'] = { flags: [], subgroups: {} };
            }
            tree['_ungrouped'].flags.push(flag);
            return;
        }

        if (parts.length === 2) {
            // Two levels - group by first level if there are multiple flags with same prefix
            const firstLevel = parts[0];
            if (tempByFirstLevel[firstLevel].length > 1) {
                // Multiple flags, create a group
                if (!tree[firstLevel]) {
                    tree[firstLevel] = { flags: [], subgroups: {} };
                }
                tree[firstLevel].flags.push(flag);
            } else {
                // Single flag, leave ungrouped
                if (!tree['_ungrouped']) {
                    tree['_ungrouped'] = { flags: [], subgroups: {} };
                }
                tree['_ungrouped'].flags.push(flag);
            }
            return;
        }

        // 3+ levels: Group by first 2 levels (e.g., "trainer.losses")
        const level1 = parts.slice(0, 2).join('.');
        if (!tree[level1]) {
            tree[level1] = { flags: [], subgroups: {} };
        }

        if (parts.length === 3) {
            // Just 3 parts, no subgroup needed
            tree[level1].flags.push(flag);
        } else {
            // 4+ parts, check if we should create a subgroup
            const level2 = parts[2];
            if (!tree[level1].subgroups[level2]) {
                tree[level1].subgroups[level2] = [];
            }
            tree[level1].subgroups[level2].push(flag);
        }
    });

    return tree;
}

function stripEnabledSuffix(flagName) {
    // Remove .enabled suffix to save space
    return flagName.replace(/\.enabled$/, '');
}

function buildFlagTree(flagColumns) {
    // Build a tree structure where each node can have children
    // Returns: { nodes: [{label, fullPath, children, colSpan}], maxDepth }
    const root = { children: {} };
    let maxDepth = 0;

    flagColumns.forEach(flag => {
        const parts = flag.split('.');
        maxDepth = Math.max(maxDepth, parts.length);

        let current = root;
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            if (!current.children[part]) {
                current.children[part] = {
                    label: part,
                    fullPath: parts.slice(0, i + 1).join('.'),
                    children: {},
                    isLeaf: i === parts.length - 1
                };
            }
            current = current.children[part];
        }
    });

    return { root, maxDepth };
}

function renderHeaderLevel(nodes, currentDepth, maxDepth, rows) {
    if (!rows[currentDepth]) {
        rows[currentDepth] = [];
    }

    nodes.forEach(node => {
        const childCount = Object.keys(node.children).length;

        if (childCount === 0) {
            // Leaf node - spans remaining rows
            const rowspan = maxDepth - currentDepth;
            const displayName = stripEnabledSuffix(node.label);
            rows[currentDepth].push({
                html: `<th class="col-flag" rowspan="${rowspan}" title="${node.fullPath}">${displayName}</th>`,
                colspan: 1
            });
        } else {
            // Internal node - calculate colspan and recurse
            const childArray = Object.values(node.children).sort((a, b) => a.label.localeCompare(b.label));
            const colspan = countLeaves(node);

            const displayName = stripEnabledSuffix(node.label);
            rows[currentDepth].push({
                html: `<th class="col-flag-group" colspan="${colspan}" title="${node.fullPath}">${displayName}</th>`,
                colspan: colspan
            });

            // Recurse for children
            renderHeaderLevel(childArray, currentDepth + 1, maxDepth, rows);
        }
    });
}

function countLeaves(node) {
    if (Object.keys(node.children).length === 0) {
        return 1;
    }

    let count = 0;
    Object.values(node.children).forEach(child => {
        count += countLeaves(child);
    });
    return count;
}

function getLeafColumnsFromTree(tree) {
    const leaves = [];

    function traverse(node) {
        if (Object.keys(node.children).length === 0) {
            // Leaf node - add its full path
            leaves.push(node.fullPath);
        } else {
            // Internal node - recurse through sorted children (same order as renderHeaderLevel)
            const childArray = Object.values(node.children).sort((a, b) => a.label.localeCompare(b.label));
            childArray.forEach(child => traverse(child));
        }
    }

    // Start with top-level nodes, sorted (same as in updateTableHeader)
    const topLevelNodes = Object.values(tree.children).sort((a, b) => a.label.localeCompare(b.label));
    topLevelNodes.forEach(node => traverse(node));

    return leaves;
}

function updateTableHeader(flagColumns) {
    const thead = document.getElementById('experiments-thead');

    // Check if all experiments are selected
    const allSelected = selectedExperiments.size === allExperiments.length && allExperiments.length > 0;
    const someSelected = selectedExperiments.size > 0 && selectedExperiments.size < allExperiments.length;
    const checkedAttr = allSelected ? 'checked' : '';

    if (flagColumns.length === 0) {
        // No flags, simple header
        thead.innerHTML = `
            <tr>
                <th class="col-drag" title="Drag to reorder">⋮⋮</th>
                <th class="col-checkbox"><input type="checkbox" id="select-all" ${checkedAttr} onchange="toggleSelectAll(this.checked)"></th>
                <th class="col-expand"><span style="display: inline-block; cursor: pointer; padding: 2px 6px; border-radius: 3px; background-color: transparent; border: 2px solid #4CAF50; color: #4CAF50; font-size: 10px; font-weight: bold;" onclick="quickCreateExperiment()" title="Create new experiment">+</span></th>
                <th class="col-id">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span>ID</span>
                        <div class="bulk-actions" id="bulk-actions" style="display: none;">
                            <button onclick="bulkStart()" class="btn-pill btn-pill-primary">Start</button>
                            <button onclick="bulkStop()" class="btn-pill">Stop</button>
                            <button onclick="bulkDuplicate()" class="btn-pill">Duplicate</button>
                            <button onclick="bulkDelete()" class="btn-pill btn-pill-danger">Delete</button>
                        </div>
                    </div>
                </th>
                <th class="col-state">State</th>
                <th class="col-epoch">Epoch</th>
                <th class="col-resources">Resources</th>
                <th class="col-padding"></th>
            </tr>
        `;
        // Set indeterminate state after innerHTML (can't be done via HTML attribute)
        const selectAll = document.getElementById('select-all');
        if (selectAll) selectAll.indeterminate = someSelected;
        return [];  // Return empty array for no columns
    }

    // Build tree and determine depth
    const { root, maxDepth } = buildFlagTree(flagColumns);
    const totalRows = maxDepth; // Number of header rows (base columns span all of them)

    // Initialize row array
    const rows = [];
    for (let i = 0; i < maxDepth; i++) {
        rows[i] = [];
    }

    // Build flag headers
    const topLevelNodes = Object.values(root.children).sort((a, b) => a.label.localeCompare(b.label));
    renderHeaderLevel(topLevelNodes, 0, maxDepth, rows);

    // Build final HTML
    let html = '';

    // First row with base columns
    html += '<tr>';
    html += `<th class="col-drag" rowspan="${totalRows}" title="Drag to reorder">⋮⋮</th>`;
    html += `<th class="col-checkbox" rowspan="${totalRows}"><input type="checkbox" id="select-all" ${checkedAttr} onchange="toggleSelectAll(this.checked)"></th>`;
    html += `<th class="col-expand" rowspan="${totalRows}"><span style="display: inline-block; cursor: pointer; padding: 2px 6px; border-radius: 3px; background-color: transparent; border: 2px solid #4CAF50; color: #4CAF50; font-size: 10px; font-weight: bold;" onclick="quickCreateExperiment()" title="Create new experiment">+</span></th>`;
    html += `<th class="col-id" rowspan="${totalRows}">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>ID</span>
            <div class="bulk-actions" id="bulk-actions" style="display: none;">
                <button onclick="bulkStart()" class="btn-pill btn-pill-primary">Start</button>
                <button onclick="bulkStop()" class="btn-pill">Stop</button>
                <button onclick="bulkDuplicate()" class="btn-pill">Duplicate</button>
                <button onclick="bulkDelete()" class="btn-pill btn-pill-danger">Delete</button>
            </div>
        </div>
    </th>`;
    html += `<th class="col-state" rowspan="${totalRows}">State</th>`;
    html += `<th class="col-epoch" rowspan="${totalRows}">Epoch</th>`;

    // Add first row of flag headers
    if (rows[0]) {
        rows[0].forEach(cell => html += cell.html);
    }
    html += `<th class="col-resources" rowspan="${totalRows}">Resources</th>`;
    html += `<th class="col-padding" rowspan="${totalRows}"></th>`;
    html += '</tr>';

    // Add remaining rows
    for (let i = 1; i < maxDepth; i++) {
        html += '<tr>';
        if (rows[i]) {
            rows[i].forEach(cell => html += cell.html);
        }
        html += '</tr>';
    }

    thead.innerHTML = html;

    // Set indeterminate state after innerHTML (can't be done via HTML attribute)
    const selectAll = document.getElementById('select-all');
    if (selectAll) selectAll.indeterminate = someSelected;

    // Return the actual column order from the tree
    return getLeafColumnsFromTree(root);
}

function toggleSelectAll(checked) {
    const selectAll = document.getElementById('select-all');

    selectedExperiments.clear();
    if (checked) {
        allExperiments.forEach(exp => selectedExperiments.add(exp.id));
    }

    // Clear indeterminate state
    if (selectAll) {
        selectAll.indeterminate = false;
    }

    updateExperimentsTable(allExperiments);
    updateBulkActions();
}

function toggleExperimentSelection(experimentId, checked) {
    if (checked) {
        selectedExperiments.add(experimentId);
    } else {
        selectedExperiments.delete(experimentId);
    }
    updateBulkActions();

    // Update select-all checkbox state
    const selectAll = document.getElementById('select-all');
    if (selectAll) {
        const allSelected = selectedExperiments.size === allExperiments.length && allExperiments.length > 0;
        const someSelected = selectedExperiments.size > 0 && selectedExperiments.size < allExperiments.length;

        selectAll.checked = allSelected;
        selectAll.indeterminate = someSelected;
    }
}

function updateBulkActions() {
    const bulkActions = document.getElementById('bulk-actions');
    if (bulkActions) {
        bulkActions.style.display = selectedExperiments.size > 0 ? 'flex' : 'none';
    }
}

function truncateFlag(flag) {
    // Truncate long flag names, show last part
    const parts = flag.split('.');
    if (parts.length > 2) {
        return '...' + parts.slice(-2).join('.');
    }
    return flag;
}

function createMainRow(exp, flagColumns) {
    const row = document.createElement('tr');
    row.className = 'main-row';
    row.dataset.expId = exp.id;
    row.draggable = true;

    const isExpanded = expandedRows.has(exp.id);

    const isRunning = exp.desired_state === 'RUNNING';
    const isSelected = selectedExperiments.has(exp.id);

    row.innerHTML = `
        <td class="col-drag drag-handle" style="cursor: grab; text-align: center; opacity: 0.5;">⋮⋮</td>
        <td class="col-checkbox">
            <input type="checkbox" class="row-checkbox" ${isSelected ? 'checked' : ''} onchange="toggleExperimentSelection('${exp.id}', this.checked); event.stopPropagation();" onclick="event.stopPropagation();">
        </td>
        <td class="col-expand">
            <span class="expand-icon ${isExpanded ? 'expanded' : ''}">▶</span>
        </td>
        <td class="col-id">
            <button class="star-btn ${exp.starred ? 'starred' : ''}" onclick="event.stopPropagation(); toggleStar('${exp.id}'); return false;" title="${exp.starred ? 'Unstar' : 'Star'}">★</button>
            <span class="wandb-link" style="cursor: pointer;" onclick="handleLinkPill('https://wandb.ai/metta-research/metta/runs/${exp.id}', event); return false;" title="Click to copy, Cmd+Click to open">W&B</span>
            <a href="https://app.datadoghq.com/logs?query=metta_run_id%3A%22${exp.id}%22" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in Datadog">log</a>
            <span style="display: inline-block; cursor: pointer;" onclick="copyToClipboard('${exp.id.replace(/'/g, "\\'")}', event); return false;" title="Click to copy ID">${exp.id}</span>
        </td>
        <td class="col-state" id="state-${exp.id}" onclick="toggleStateEdit('${exp.id}'); event.stopPropagation();" style="cursor: pointer;">
            <span id="state-view-${exp.id}">
                ${formatStateTransition(exp.current_state, exp.desired_state)}
            </span>
            <span id="state-edit-${exp.id}" style="display: none; gap: 2px; white-space: nowrap;">
                <span class="state-pill stopped" onclick="setDesiredStateAndClose('${exp.id}', 'STOPPED'); event.stopPropagation();" title="Stop" style="cursor: pointer; padding: 2px 6px; border-radius: 8px; background-color: #9E9E9E; color: white; font-size: 12px;">■</span>
                <span style="color: #666; font-size: 12px;">|</span>
                <span class="state-pill running" onclick="setDesiredStateAndClose('${exp.id}', 'RUNNING'); event.stopPropagation();" title="Start" style="cursor: pointer; padding: 2px 6px; border-radius: 8px; background-color: #4CAF50; color: white; font-size: 12px;">▶</span>
            </span>
        </td>
        <td class="col-epoch">${exp.latest_epoch !== null && exp.latest_epoch !== undefined ? exp.latest_epoch : '—'}</td>
        ${flagColumns.map(flag => {
            const value = exp.flags[flag];
            return `<td class="col-flag">${formatFlagValue(value)}</td>`;
        }).join('')}
        <td class="col-resources">${exp.nodes}×${exp.gpus}</td>
        <td class="col-padding"></td>
    `;

    // Click anywhere on row to expand/collapse
    row.addEventListener('click', (e) => {
        toggleRow(exp.id);
    });

    // Drag and drop event handlers
    row.addEventListener('dragstart', (e) => {
        draggedRow = row;
        row.style.opacity = '0.5';
        e.dataTransfer.effectAllowed = 'move';
    });

    row.addEventListener('dragend', (e) => {
        // Remove all drag-over indicators
        const tbody = row.parentNode;
        tbody.querySelectorAll('.drag-over-before, .drag-over-after').forEach(r => {
            r.classList.remove('drag-over-before', 'drag-over-after');
        });
        row.style.opacity = '1';
        draggedRow = null;
    });

    row.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (draggedRow && draggedRow !== row) {
            const tbody = row.parentNode;
            const draggedIndex = Array.from(tbody.children).indexOf(draggedRow);
            const targetIndex = Array.from(tbody.children).indexOf(row);

            // Remove all drag-over indicators
            tbody.querySelectorAll('.drag-over-before, .drag-over-after').forEach(r => {
                r.classList.remove('drag-over-before', 'drag-over-after');
            });

            // Get the expanded row for the dragged row (next sibling)
            const draggedExpandedRow = draggedRow.nextElementSibling;
            const isExpandedRow = draggedExpandedRow && draggedExpandedRow.classList.contains('expanded-row');

            if (draggedIndex < targetIndex) {
                // Moving down: show indicator after target
                row.classList.add('drag-over-after');

                // Insert after target's expanded row
                const targetExpandedRow = row.nextElementSibling;
                if (targetExpandedRow && targetExpandedRow.classList.contains('expanded-row')) {
                    targetExpandedRow.after(draggedRow);
                } else {
                    row.after(draggedRow);
                }
                if (isExpandedRow) {
                    draggedRow.after(draggedExpandedRow);
                }
            } else {
                // Moving up: show indicator before target
                row.classList.add('drag-over-before');

                // Insert before target
                row.before(draggedRow);
                if (isExpandedRow) {
                    draggedRow.after(draggedExpandedRow);
                }
            }
        }
    });

    row.addEventListener('drop', async (e) => {
        e.preventDefault();
        if (draggedRow) {
            const tbody = row.parentNode;

            // Remove all drag-over indicators
            tbody.querySelectorAll('.drag-over-before, .drag-over-after').forEach(r => {
                r.classList.remove('drag-over-before', 'drag-over-after');
            });

            // Save new order to database
            const newOrder = Array.from(tbody.children)
                .filter(r => r.classList.contains('main-row'))
                .map(r => r.dataset.expId);
            await saveExperimentOrder(newOrder);
        }
    });

    return row;
}

function abbreviateStatus(status) {
    const statusMap = {
        'running': 'R',
        'stopped': 'S',
        'pending': 'P',
        'failed': 'F',
        'succeeded': 'X',
        'init': 'I',
        'terminated': 'T',
        'cancelled': 'C',
        'unknown': '?'
    };
    const lower = status.toLowerCase();
    return statusMap[lower] || status.charAt(0).toUpperCase();
}

function formatStateTransition(current, desired) {
    const currentLower = current.toLowerCase();
    const desiredLower = desired.toLowerCase();

    const currentAbbrev = abbreviateStatus(currentLower);
    const desiredAbbrev = abbreviateStatus(desiredLower);

    // Capitalize first letter for tooltip
    const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
    const currentTitle = capitalize(current);
    const desiredTitle = capitalize(desired);

    // If current and desired are the same, show single state
    if (currentLower === desiredLower) {
        return `<span class="status-badge ${currentLower}" title="${currentTitle}">${currentAbbrev}</span>`;
    }

    // Treat FAILED as effectively STOPPED - if desired is STOPPED and current is FAILED, just show F
    if (currentLower === 'failed' && desiredLower === 'stopped') {
        return `<span class="status-badge ${currentLower}" title="${currentTitle}">${currentAbbrev}</span>`;
    }

    // Otherwise show transition: current->desired
    return `<span class="status-badge ${currentLower}" title="${currentTitle}">${currentAbbrev}</span> → <span class="status-badge ${desiredLower}" title="${desiredTitle}">${desiredAbbrev}</span>`;
}

function formatFlagValue(value) {
    if (value === undefined || value === null) {
        return '<span class="flag-value empty">-</span>';
    }
    if (typeof value === 'boolean') {
        const className = value ? 'boolean-true' : 'boolean-false';
        const emoji = value ? '✓' : '✗';
        return `<span class="flag-value ${className}" style="font-size: 200%; display: block; text-align: center;">${emoji}</span>`;
    }
    return `<span class="flag-value">${value}</span>`;
}

function createExpandedRow(exp, numFlagColumns) {
    const row = document.createElement('tr');
    row.className = 'expanded-row';
    if (expandedRows.has(exp.id)) {
        row.classList.add('show');
    }
    row.dataset.expId = exp.id;

    const totalColumns = 8 + numFlagColumns;  // drag, checkbox, expand, id, state, epoch, resources, padding + flags

    // Build flags table (sorted alphabetically)
    const flagsHtml = Object.entries(exp.flags || {}).length > 0
        ? Object.entries(exp.flags || {})
            .sort((a, b) => a[0].localeCompare(b[0]))  // Sort by key
            .map(([key, val]) => `
                <tr class="flag-row" data-flag-key="${key}">
                    <td class="flag-key" style="padding: 4px 8px 4px 0; font-size: 12px; color: #666; border-bottom: 1px solid #eee; white-space: nowrap;">${key}</td>
                    <td class="flag-value-cell" style="padding: 4px 0 4px 12px; font-size: 12px; border-bottom: 1px solid #eee; white-space: nowrap;">${val}</td>
                    <td class="flag-actions" style="padding: 4px 0 4px 8px; font-size: 12px; border-bottom: 1px solid #eee; display: none;"></td>
                </tr>
            `).join('')
        : '<tr><td colspan="3" style="padding: 8px 0; font-size: 12px; color: #999;">No flags</td></tr>';

    const fullCommand = buildCommand(exp);
    const escapedCommand = fullCommand.replace(/'/g, "\\'").replace(/"/g, '&quot;');

    row.innerHTML = `
        <td colspan="${totalColumns}">
            <div class="expanded-details">
                <!-- First Row: Configuration, Jobs, Checkpoints -->
                <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 20px;">
                    <div class="detail-section" style="width: auto; min-width: 400px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="display: flex; align-items: center; gap: 8px;">
                                Configuration
                                <span id="config-buttons-${exp.id}">
                                    ${editingRows.has(exp.id) ? `
                                        <a onclick="toggleConfigEdit('${exp.id}')" id="config-edit-btn-${exp.id}" class="wandb-link" style="cursor: pointer; margin-left: 0; margin-right: 6px; font-size: 12px; color: #4caf50;" title="Save changes">✓</a>
                                        <a onclick="cancelConfigEdit('${exp.id}')" class="wandb-link" style="cursor: pointer; margin-left: 0; font-size: 12px; color: #f44336;" title="Cancel">✗</a>
                                    ` : `
                                        <a onclick="toggleConfigEdit('${exp.id}')" id="config-edit-btn-${exp.id}" class="wandb-link" style="cursor: pointer; margin-left: 0; font-size: 11px;" title="Edit configuration">✎</a>
                                    `}
                                </span>
                            </h3>
                        </div>
                        <div class="detail-grid" id="config-grid-${exp.id}">
                            <span class="detail-label">Experiment ID:</span>
                            <span class="detail-value">
                                <span class="config-field" data-field="id" data-value="${exp.id || ''}">${exp.id || '<span style="color: #f44336;">[required]</span>'}</span>
                                <a href="https://wandb.ai/metta-research/metta/runs/${exp.id}" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in W&B">W&B</a>
                                <a href="https://app.datadoghq.com/logs?query=metta_run_id%3A%22${exp.id}%22" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in Datadog">log</a>
                            </span>

                            <span class="detail-label">Tool Path:</span>
                            <span class="detail-value config-field" data-field="tool_path" data-value="${exp.tool_path || ''}">${exp.tool_path || '<span style="color: #f44336;">[required]</span>'}</span>

                            <span class="detail-label">Nodes x GPUs:</span>
                            <span class="detail-value">
                                <span class="config-field" data-field="nodes" data-value="${exp.nodes || ''}">${exp.nodes || '<span style="color: #f44336;">[required]</span>'}</span> x <span class="config-field" data-field="gpus" data-value="${exp.gpus || ''}">${exp.gpus || '<span style="color: #f44336;">[required]</span>'}</span>
                            </span>

                            <span class="detail-label">Git Branch:</span>
                            <span class="detail-value config-field" data-field="git_branch" data-value="${exp.git_branch || ''}">${exp.git_branch || '-'}</span>
                        </div>

                        ${exp.description ? `<p style="margin-top: 10px; font-size: 12px; color: #666;">${exp.description}</p>` : ''}

                        <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">

                        <table id="flags-table-${exp.id}" style="width: auto; border-collapse: collapse;">
                            ${flagsHtml}
                        </table>
                        <div id="add-flag-btn-${exp.id}" style="display: none; margin-top: 10px;">
                            <button class="btn btn-small" onclick="addNewFlag('${exp.id}')">+ Add Flag</button>
                        </div>
                    </div>

                    <div class="detail-section" style="width: auto; flex-shrink: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h3 style="margin: 0;">Jobs</h3>
                        </div>
                        ${ScrollablePanel.createContainer(`jobs-${exp.id}`, '<p style="color: #999; font-size: 12px;">Loading...</p>')}
                    </div>

                    <!-- Checkpoints Panel -->
                    <div class="detail-section" style="flex-shrink: 0;">
                        <h3>Checkpoints</h3>
                        <div id="checkpoints-${exp.id}" class="checkpoints-list" style="max-height: 300px; overflow-y: auto;">
                            <p style="color: #999; font-size: 12px;">Loading...</p>
                        </div>
                    </div>
                </div>

                <!-- Second Row: Command Panel -->
                <div style="display: flex;">
                    <div class="detail-section" style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <h3 style="margin: 0;">Command</h3>
                            <button class="copy-btn copy-btn-left" onclick="copyToClipboard('${escapedCommand}', event); return false;" title="Copy command">⎘</button>
                        </div>
                        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-family: 'Monaco', 'Menlo', 'Courier New', monospace; font-size: 12px; overflow-x: auto; text-align: left; white-space: pre-wrap; word-break: break-word;">${fullCommand}</div>
                    </div>
                </div>
            </div>
        </td>
    `;

    return row;
}

function formatFlags(flags) {
    return Object.entries(flags || {})
        .map(([key, val]) => `${key}=${val}`)
        .join(' ');
}

function buildCommand(exp) {
    const parts = [exp.base_command];

    // Always add GPUs and nodes as CLI flags
    parts.push(`--gpus=${exp.gpus}`);
    parts.push(`--nodes=${exp.nodes}`);

    // Add git branch if specified (skip invalid values like "-")
    if (exp.git_branch && exp.git_branch !== '-') {
        parts.push(`--git-ref=${exp.git_branch}`);
    }

    // Add tool path
    if (exp.tool_path) {
        parts.push(exp.tool_path);
    }

    // Always add run name using experiment ID
    parts.push(`run=${exp.id}`);

    // Add all flags in sorted order
    const sortedFlags = Object.entries(exp.flags || {}).sort((a, b) => a[0].localeCompare(b[0]));
    for (const [key, value] of sortedFlags) {
        if (typeof value === 'boolean') {
            parts.push(`${key}=${value.toString().toLowerCase()}`);
        } else if (typeof value === 'string') {
            if (value.includes(' ')) {
                parts.push(`${key}="${value}"`);
            } else {
                parts.push(`${key}=${value}`);
            }
        } else {
            parts.push(`${key}=${value}`);
        }
    }

    return parts.join(' ');
}

async function toggleRow(experimentId) {
    const mainRow = document.querySelector(`tr.main-row[data-exp-id="${experimentId}"]`);
    const expandedRow = document.querySelector(`tr.expanded-row[data-exp-id="${experimentId}"]`);
    const icon = mainRow.querySelector('.expand-icon');

    const isExpanding = !expandedRows.has(experimentId);

    if (isExpanding) {
        // Expand
        expandedRows.add(experimentId);
        expandedRow.classList.add('show');
        icon.classList.add('expanded');

        // Load job history and checkpoints
        await loadJobHistory(experimentId);
        await loadCheckpoints(experimentId);
    } else {
        // Collapse
        expandedRows.delete(experimentId);
        expandedRow.classList.remove('show');
        icon.classList.remove('expanded');
    }

    // Persist to database
    try {
        await apiCall(`/experiments/${experimentId}/expanded`, {
            method: 'POST',
            body: JSON.stringify({ is_expanded: isExpanding })
        });
    } catch (error) {
        console.error('Error saving expanded state:', error);
    }
}

async function toggleStar(experimentId) {
    try {
        // Get current state
        const exp = allExperiments.find(e => e.id === experimentId);
        if (!exp) return;

        const newStarredState = !exp.starred;

        // Update server
        await apiCall(`/experiments/${experimentId}/starred`, {
            method: 'POST',
            body: JSON.stringify({ starred: newStarredState })
        });

        // Update local state
        exp.starred = newStarredState;

        // Update UI button
        const starBtn = document.querySelector(`tr.main-row[data-exp-id="${experimentId}"] .star-btn`);
        if (starBtn) {
            if (newStarredState) {
                starBtn.classList.add('starred');
                starBtn.title = 'Unstar';
            } else {
                starBtn.classList.remove('starred');
                starBtn.title = 'Star';
            }
        }
    } catch (error) {
        console.error('Error toggling star:', error);
        showNotification('Failed to toggle star', 'error');
    }
}

async function loadJobHistory(experimentId) {
    const startBtn = document.getElementById(`start-btn-${experimentId}`);
    const stopBtn = document.getElementById(`stop-btn-${experimentId}`);

    // Don't update if row is being edited
    if (editingRows.has(experimentId)) {
        return;
    }

    // Get or create ScrollablePanel for this experiment's jobs
    if (!jobsPanels.has(experimentId)) {
        jobsPanels.set(experimentId, new ScrollablePanel(`jobs-${experimentId}`));
    }
    const panel = jobsPanels.get(experimentId);

    // Save current scroll position before updating content
    panel.saveScrollPosition();

    try{
        // Get experiment to find current job
        const expData = await apiCall(`/experiments/${experimentId}`);
        const experiment = expData.experiment;

        // Get job history
        const data = await apiCall(`/experiments/${experimentId}/jobs?limit=10`);
        const jobs = data.jobs || [];

        // Find current job
        const currentJobId = experiment.current_job_id;
        const currentJob = jobs.find(j => j.id === currentJobId);
        const hasRunningJob = currentJob && (currentJob.status === 'RUNNING' || currentJob.status === 'PENDING');

        // Update button states
        if (startBtn) startBtn.disabled = hasRunningJob;
        if (stopBtn) stopBtn.disabled = !hasRunningJob;

        // Show all jobs in table format
        const rows = jobs.map(job => {
            // Extract numeric ID from full ID (e.g., "daveey.ca4.4x4.mcl_1.0-9482" -> "9482")
            const numericId = job.id.split('-').pop();
            return `
                <tr>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; white-space: nowrap; cursor: pointer;" onclick="event.stopPropagation(); copyToClipboard('${job.id}', event); return false;" title="Click to copy full ID">${numericId}</td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; white-space: nowrap;"><span class="status-badge ${job.status.toLowerCase()}" title="${job.status}">${abbreviateStatus(job.status)}</span></td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; white-space: nowrap;"><a href="https://skypilot-api.softmax-research.net/dashboard/jobs/${job.id}" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in SkyPilot Dashboard">sky</a> <a href="https://app.datadoghq.com/logs?query=skypilot_task_id%3A%2A${job.id}%2A%20metta_run_id%3A%22${job.experiment_id}%22" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in Datadog">log</a></td>
                </tr>
            `;
        }).join('');

        const html = `
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Job ID</th>
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Status</th>
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Links</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;

        // Update panel content with automatic scroll preservation
        panel.update(html);
    } catch (error) {
        panel.update('<p style="color: #f44336; font-size: 12px;">Error loading jobs</p>');
        console.error('Error loading job history:', error);
    }
}

async function loadCheckpoints(experimentId) {
    const container = document.getElementById(`checkpoints-${experimentId}`);

    if (!container) {
        console.warn(`Checkpoints container not found for experiment ${experimentId}`);
        return;
    }

    // Save scroll position before updating
    const scrollTop = container.scrollTop;

    try {
        const data = await apiCall(`/experiments/${experimentId}/checkpoints?limit=20`);
        const checkpoints = data.checkpoints || [];

        if (checkpoints.length === 0) {
            container.innerHTML = '<p style="color: #999; font-size: 12px;">No checkpoints yet</p>';
            return;
        }

        // Build pills for each storage location
        const pillStyle = 'display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; cursor: pointer; margin-right: 4px; transition: opacity 0.2s;';

        const rows = checkpoints.map(cp => {
            const replayCount = cp.replay_paths ? cp.replay_paths.length : 0;

            // Build URLs for different storage locations
            const s3Url = cp.model_path || '';
            const mettaUrl = cp.version ? `metta://policies/${cp.version}` : `metta://policy/${experimentId}`;

            const s3Pill = s3Url
                ? `<span class="storage-pill" style="${pillStyle} background-color: #E8F5E9; color: #2E7D32;" onclick="handleLinkPill('${s3Url.replace(/'/g, "\\'")}', event); return false;" title="Click to copy, Cmd+Click to open">s3</span>`
                : '';

            const mettaPill = `<span class="storage-pill" style="${pillStyle} background-color: #E3F2FD; color: #1565C0;" onclick="handleLinkPill('${mettaUrl.replace(/'/g, "\\'")}', event); return false;" title="Click to copy, Cmd+Click to open">metta</span>`;

            const obsUrl = `https://api.observatory.softmax-research.net/stats/policies?name_fuzzy=${experimentId}&limit=500`;
            const obsPill = `<span class="storage-pill" style="${pillStyle} background-color: #F3E5F5; color: #6A1B9A;" onclick="handleLinkPill('${obsUrl.replace(/'/g, "\\'")}', event); return false;" title="Click to copy, Cmd+Click to open">obs</span>`;

            const policyVersion = cp.policy_version || '-';

            return `
                <tr>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; white-space: nowrap;">${cp.epoch}</td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; color: #666; white-space: nowrap;">${formatDuration(cp.created_at)}</td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; color: #666; white-space: nowrap;">${policyVersion}</td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; white-space: nowrap;">${s3Pill}${mettaPill}${obsPill}</td>
                    <td style="padding: 4px 6px; font-size: 11px; border-bottom: 1px solid #eee; text-align: center; white-space: nowrap;">${replayCount}</td>
                </tr>
            `;
        }).join('');

        const html = `
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Epoch</th>
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Age</th>
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Version</th>
                        <th style="padding: 4px 6px; text-align: left; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Links</th>
                        <th style="padding: 4px 6px; text-align: center; font-weight: 600; border-bottom: 2px solid #e0e0e0; font-size: 10px; white-space: nowrap;">Replays</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;

        container.innerHTML = html;

        // Restore scroll position after updating
        container.scrollTop = scrollTop;
    } catch (error) {
        container.innerHTML = '<p style="color: #f44336; font-size: 12px;">Error loading checkpoints</p>';
        console.error('Error loading checkpoints:', error);
    }
}


async function startExperiment(experimentId, event) {
    if (event) event.stopPropagation();

    try {
        await apiCall(`/experiments/${experimentId}/state`, {
            method: 'POST',
            body: JSON.stringify({ desired_state: 'RUNNING' })
        });
        showNotification('Experiment started', 'success');
        await loadData();

        // Reload job history for this experiment
        await loadJobHistory(experimentId);
    } catch (error) {
        showNotification('Error starting experiment', 'error');
        console.error('Error starting experiment:', error);
    }
}

async function stopExperiment(experimentId, event) {
    if (event) event.stopPropagation();

    try {
        await apiCall(`/experiments/${experimentId}/state`, {
            method: 'POST',
            body: JSON.stringify({ desired_state: 'STOPPED' })
        });
        showNotification('Experiment stopped', 'success');
        await loadData();

        // Reload job history for this experiment
        await loadJobHistory(experimentId);
    } catch (error) {
        showNotification('Error stopping experiment', 'error');
        console.error('Error stopping experiment:', error);
    }
}

async function stopJob(jobId, event) {
    if (event) event.stopPropagation();

    if (!confirm(`Stop job ${jobId}?`)) return;

    try {
        await apiCall(`/jobs/${jobId}/cancel`, { method: 'POST' });
        showNotification(`Job ${jobId} stopped`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error stopping job:', error);
    }
}

async function cancelConfigEdit(experimentId) {
    // Remove from editing set
    editingRows.delete(experimentId);

    // Reload data to restore original values - this will rebuild the row in non-edit mode
    await loadData();
}

async function toggleConfigEdit(experimentId) {
    const btnContainer = document.getElementById(`config-buttons-${experimentId}`);
    const btn = document.getElementById(`config-edit-btn-${experimentId}`);
    const grid = document.getElementById(`config-grid-${experimentId}`);
    const fields = grid.querySelectorAll('.config-field');
    const flagsTable = document.getElementById(`flags-table-${experimentId}`);
    const addFlagBtn = document.getElementById(`add-flag-btn-${experimentId}`);

    if (btn.textContent === '✎') {
        // Enter edit mode
        // Add Save and Cancel icons
        btnContainer.innerHTML = `
            <a onclick="toggleConfigEdit('${experimentId}')" id="config-edit-btn-${experimentId}" class="wandb-link" style="cursor: pointer; margin-left: 0; margin-right: 6px; font-size: 12px; color: #4caf50;" title="Save changes">✓</a>
            <a onclick="cancelConfigEdit('${experimentId}')" class="wandb-link" style="cursor: pointer; margin-left: 0; font-size: 12px; color: #f44336;" title="Cancel">✗</a>
        `;
        editingRows.add(experimentId);  // Track that this row is being edited

        // Make config fields editable
        fields.forEach(field => {
            // Use data-value if available (for required fields), otherwise use textContent
            const currentValue = field.dataset.value !== undefined
                ? field.dataset.value
                : (field.textContent === '-' ? '' : field.textContent);
            field.dataset.originalValue = currentValue;
            field.textContent = currentValue; // Clear '-' or [required] from display

            field.contentEditable = 'true';
            field.style.backgroundColor = '#fff8e1';
            field.style.cursor = 'text';
            field.style.padding = '4px';
            field.style.borderRadius = '2px';
        });

        // Make flags editable
        const flagRows = flagsTable.querySelectorAll('.flag-row');
        flagRows.forEach(row => {
            const keyCell = row.querySelector('.flag-key');
            const valueCell = row.querySelector('.flag-value-cell');
            const actionsCell = row.querySelector('.flag-actions');

            // Store original values
            row.dataset.originalKey = keyCell.textContent;
            row.dataset.originalValue = valueCell.textContent;

            // Make cells contenteditable
            keyCell.contentEditable = 'true';
            keyCell.style.backgroundColor = '#fff8e1';
            keyCell.style.cursor = 'text';

            valueCell.contentEditable = 'true';
            valueCell.style.backgroundColor = '#fff8e1';
            valueCell.style.cursor = 'text';

            // Add delete button
            actionsCell.style.display = '';
            actionsCell.innerHTML = `<button class="btn btn-small btn-danger" onclick="removeFlagRow(this)" style="padding: 2px 6px; font-size: 11px;">×</button>`;
        });

        // Show "Add Flag" button
        addFlagBtn.style.display = 'block';

    } else {
        // Save mode
        btnContainer.innerHTML = `<a class="wandb-link" style="cursor: not-allowed; margin-left: 0; font-size: 12px; color: #999;" title="Saving...">✓</a>`;
        editingRows.delete(experimentId);  // Remove from editing tracking

        try {
            // Collect updated config values
            const updates = {};
            fields.forEach(field => {
                const fieldName = field.dataset.field;
                const newValue = field.textContent.trim() || null;
                updates[fieldName] = newValue;
            });

            // Collect flags
            const flags = {};
            const flagRows = flagsTable.querySelectorAll('.flag-row');
            flagRows.forEach(row => {
                const keyCell = row.querySelector('.flag-key');
                const valueCell = row.querySelector('.flag-value-cell');

                if (keyCell && valueCell) {
                    const key = keyCell.textContent.trim();
                    let value = valueCell.textContent.trim();

                    if (key) {
                        // Try to parse value as JSON for proper type
                        try {
                            value = JSON.parse(value);
                        } catch {
                            // Keep as string if not valid JSON
                        }
                        flags[key] = value;
                    }
                }
            });

            // Send config update to server
            await apiCall(`/experiments/${experimentId}`, {
                method: 'PATCH',
                body: JSON.stringify(updates)
            });

            // Send flags update to server
            await apiCall(`/experiments/${experimentId}/flags`, {
                method: 'POST',
                body: JSON.stringify({ flags })
            });

            showNotification('Configuration updated', 'success');
            await loadData();
        } catch (error) {
            showNotification('Error updating configuration', 'error');
            console.error('Error updating configuration:', error);
            await loadData(); // Reload to revert changes
        }
    }
}

function addNewFlag(experimentId) {
    const flagsTable = document.getElementById(`flags-table-${experimentId}`);

    // Remove "No flags" row if it exists
    const noFlagsRow = flagsTable.querySelector('td[colspan="3"]');
    if (noFlagsRow) {
        noFlagsRow.parentElement.remove();
    }

    const newRow = document.createElement('tr');
    newRow.className = 'flag-row';
    newRow.innerHTML = `
        <td class="flag-key" contenteditable="true" style="padding: 4px 8px 4px 0; font-size: 12px; color: #666; border-bottom: 1px solid #eee; white-space: nowrap; background-color: #fff8e1; cursor: text;"></td>
        <td class="flag-value-cell" contenteditable="true" style="padding: 4px 0 4px 12px; font-size: 12px; border-bottom: 1px solid #eee; white-space: nowrap; background-color: #fff8e1; cursor: text;"></td>
        <td class="flag-actions" style="padding: 4px 0 4px 8px; font-size: 12px; border-bottom: 1px solid #eee;">
            <button class="btn btn-small btn-danger" onclick="removeFlagRow(this)" style="padding: 2px 6px; font-size: 11px;">×</button>
        </td>
    `;

    flagsTable.appendChild(newRow);

    // Focus the key cell
    const keyCell = newRow.querySelector('.flag-key');
    keyCell.focus();
}

function removeFlagRow(btn) {
    const row = btn.closest('tr');
    const table = row.closest('table');
    row.remove();

    // If no more flags, show "No flags" message
    if (table.querySelectorAll('.flag-row').length === 0) {
        const noFlagsRow = document.createElement('tr');
        noFlagsRow.innerHTML = '<td colspan="3" style="padding: 8px 0; font-size: 12px; color: #999;">No flags</td>';
        table.appendChild(noFlagsRow);
    }
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 60) {
        return `${diffMins}m ago`;
    } else if (diffMins < 1440) {
        return `${Math.floor(diffMins / 60)}h ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function formatDuration(startedAt) {
    if (!startedAt) return '-';

    const start = new Date(startedAt);
    const now = new Date();
    const diffMs = now - start;

    const hours = Math.floor(diffMs / 3600000);
    const mins = Math.floor((diffMs % 3600000) / 60000);

    if (hours > 0) {
        return `${hours}h ${mins}m`;
    } else {
        return `${mins}m`;
    }
}

function updateJobsTable(jobs) {
    const tbody = document.getElementById('jobs-tbody');

    // Apply filters
    let filteredJobs = jobs;

    // Apply orphaned filter
    if (showOrphanedOnly) {
        // Get all experiment IDs
        const experimentIds = new Set(allExperiments.map(e => e.id));
        // Filter to jobs that don't have a matching experiment
        filteredJobs = filteredJobs.filter(job => !experimentIds.has(job.experiment_id));
    }

    // Apply my jobs filter
    if (showMyJobsOnly) {
        filteredJobs = filteredJobs.filter(job =>
            job.experiment_id.startsWith(MY_USER_ID + '.')
        );
    }

    // Apply text filter
    if (jobsFilterText) {
        const searchLower = jobsFilterText.toLowerCase();
        filteredJobs = filteredJobs.filter(job =>
            job.experiment_id.toLowerCase().includes(searchLower) ||
            job.id.toString().includes(searchLower)
        );
    }

    // Compute hash to detect changes
    const jobsHash = JSON.stringify(filteredJobs.map(j => ({
        id: j.id,
        status: j.status,
        experiment_id: j.experiment_id,
        nodes: j.nodes,
        gpus: j.gpus,
        started_at: j.started_at
    }))) + selectedJobs.size;

    // Only update if data has changed
    if (jobsHash === lastJobsHash && tbody.children.length > 0) {
        return;
    }
    lastJobsHash = jobsHash;

    if (!filteredJobs || filteredJobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No jobs match the current filters</td></tr>';
        updateJobsBulkActions();
        return;
    }

    // Parse job command to extract branch and flags
    function parseJobCommand(command) {
        const result = { branch: null, flags: {}, fullCommand: command || '' };
        if (!command) return result;

        // Extract git branch (look for patterns like "--branch=main" or "main" after certain keywords)
        const branchMatch = command.match(/--(?:git[-_]?)?branch[=\s]+([^\s]+)/i) ||
                           command.match(/git checkout ([^\s;&|]+)/);
        if (branchMatch) result.branch = branchMatch[1];

        // Extract all flag-like patterns (--key=value or --key value)
        const flagMatches = [...command.matchAll(/--([a-zA-Z][-a-zA-Z0-9_]*?)(?:=|\s+)([^\s-][^\s]*?)(?=\s+--|$|\s+[^-])/g)];
        flagMatches.forEach(match => {
            result.flags[match[1]] = match[2];
        });

        return result;
    }

    const jobsHtml = filteredJobs.flatMap(job => {
        // Parse command to extract actual resource requirements
        const command = job.command || '';
        let resources = `${job.nodes}×${job.gpus}`;

        // Try to extract from command line like "--nodes=4 --gpus=4"
        const nodesMatch = command.match(/--nodes=(\d+)/);
        const gpusMatch = command.match(/--gpus=(\d+)/);
        if (nodesMatch && gpusMatch) {
            resources = `${nodesMatch[1]}×${gpusMatch[1]}`;
        }

        const isSelected = selectedJobs.has(job.id);
        const canStop = job.status.toLowerCase() === 'running' || job.status.toLowerCase() === 'pending';
        const isExpanded = expandedJobs.has(job.id);
        const parsed = parseJobCommand(command);

        const mainRow = `
            <tr class="main-row" data-job-id="${job.id}" onclick="toggleJobRow('${job.id}')">
                <td class="col-expand">
                    <span class="expand-icon ${isExpanded ? 'expanded' : ''}">▶</span>
                </td>
                <td class="col-checkbox" style="padding: 4px 2px;" onclick="event.stopPropagation();">
                    <input type="checkbox" ${canStop ? '' : 'disabled'} ${isSelected ? 'checked' : ''} onchange="toggleJobSelection('${job.id}', this.checked)">
                </td>
                <td style="font-family: monospace; font-size: 12px; padding: 4px 6px;">
                    <span onclick="copyToClipboard('${job.id}', event); return false;" style="cursor: pointer;" title="Click to copy">${job.id}</span>
                    <a href="https://wandb.ai/metta-research/metta/runs/${job.experiment_id}" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in W&B" style="margin-left: 6px;">w&b</a>
                    <a href="https://skypilot-api.softmax-research.net/dashboard/jobs/${job.id}" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in SkyPilot Dashboard" style="margin-left: 6px;">sky</a>
                    <a href="https://app.datadoghq.com/logs?query=skypilot_task_id%3A%2A${job.id}%2A%20metta_run_id%3A%22${job.experiment_id}%22" target="_blank" class="wandb-link" onclick="event.stopPropagation();" title="Open in Datadog" style="margin-left: 6px;">log</a>
                </td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; padding: 4px 6px;">
                    <span onclick="copyToClipboard('${job.experiment_id.replace(/'/g, "\\'")}', event); return false;" style="cursor: pointer;" title="Click to copy">${job.experiment_id}</span>
                </td>
                <td style="padding: 4px 6px;"><span class="status-badge ${job.status.toLowerCase()}" title="${job.status}">${abbreviateStatus(job.status)}</span></td>
                <td style="font-family: monospace; font-size: 11px; padding: 4px 6px;">${resources}</td>
                <td style="font-size: 11px; padding: 4px 6px;">${formatDuration(job.started_at)}</td>
                <td style="font-size: 11px; padding: 4px 6px;">${job.started_at ? formatTime(job.started_at) : 'Not started'}</td>
                <td></td>
            </tr>
        `;

        const flagsHtml = Object.keys(parsed.flags).length > 0
            ? Object.entries(parsed.flags).map(([key, val]) => `
                <div style="display: flex; gap: 8px; padding: 4px 0;">
                    <span style="color: #666; min-width: 120px;">${key}:</span>
                    <span style="font-family: monospace;">${val}</span>
                </div>
            `).join('')
            : '<div style="color: #999;">No flags detected</div>';

        const expandedRow = `
            <tr class="expanded-row ${isExpanded ? 'show' : ''}" data-job-id="${job.id}">
                <td colspan="9" style="padding: 12px 20px; background: #f9f9f9; border-top: none;">
                    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                        <div>
                            ${parsed.branch ? `
                                <div style="margin-bottom: 12px;">
                                    <strong style="display: block; margin-bottom: 4px;">Branch:</strong>
                                    <span style="font-family: monospace; padding: 2px 6px; background: #e8f4f8; border-radius: 3px;">${parsed.branch}</span>
                                </div>
                            ` : ''}
                            <div>
                                <strong style="display: block; margin-bottom: 4px;">Flags:</strong>
                                ${flagsHtml}
                            </div>
                        </div>
                        <div>
                            <strong style="display: block; margin-bottom: 4px;">Command:</strong>
                            <div style="background: white; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; font-size: 11px; word-break: break-all; cursor: pointer;" onclick="copyToClipboard(\`${parsed.fullCommand.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`, event); return false;" title="Click to copy">
                                ${parsed.fullCommand}
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        `;

        return [mainRow, expandedRow];
    }).flat().join('');

    tbody.innerHTML = jobsHtml;
    updateJobsBulkActions();
}

// Toggle job expansion
function toggleJobRow(jobId) {
    const mainRow = document.querySelector(`tr.main-row[data-job-id="${jobId}"]`);
    const expandedRow = document.querySelector(`tr.expanded-row[data-job-id="${jobId}"]`);
    const icon = mainRow.querySelector('.expand-icon');

    const isExpanding = !expandedJobs.has(jobId);

    if (isExpanding) {
        expandedJobs.add(jobId);
        expandedRow.classList.add('show');
        icon.classList.add('expanded');
    } else {
        expandedJobs.delete(jobId);
        expandedRow.classList.remove('show');
        icon.classList.remove('expanded');
    }
}

// Experiment actions
async function setDesiredState(experimentId, state) {
    try {
        await apiCall(`/experiments/${experimentId}/state`, {
            method: 'POST',
            body: JSON.stringify({ desired_state: state }),
        });
        await loadData();
    } catch (error) {
        console.error('Error updating state:', error);
    }
}

function toggleStateEdit(experimentId) {
    const viewSpan = document.getElementById(`state-view-${experimentId}`);
    const editSpan = document.getElementById(`state-edit-${experimentId}`);

    if (editSpan.style.display === 'none') {
        viewSpan.style.display = 'none';
        editSpan.style.display = 'inline-flex';
        stateEditingRows.add(experimentId);
    } else {
        viewSpan.style.display = 'inline';
        editSpan.style.display = 'none';
        stateEditingRows.delete(experimentId);
    }
}

async function setDesiredStateAndClose(experimentId, state) {
    await setDesiredState(experimentId, state);
    // After state is set and data is loaded, return to view mode
    const viewSpan = document.getElementById(`state-view-${experimentId}`);
    const editSpan = document.getElementById(`state-edit-${experimentId}`);
    if (viewSpan && editSpan) {
        viewSpan.style.display = 'inline';
        editSpan.style.display = 'none';
        stateEditingRows.delete(experimentId);
    }
}

async function deleteExperiment(experimentId) {
    try {
        await apiCall(`/experiments/${experimentId}`, {
            method: 'DELETE',
        });
        expandedRows.delete(experimentId);

        // Show notification with undo button
        showNotification(`Deleted experiment ${experimentId}`, 'success', async () => {
            try {
                await apiCall(`/experiments/${experimentId}/undelete`, {
                    method: 'POST',
                });
                showNotification(`Restored experiment ${experimentId}`, 'success');
                await loadData();
            } catch (error) {
                console.error('Error restoring experiment:', error);
                showNotification(`Failed to restore experiment`, 'error');
            }
        });

        await loadData();
    } catch (error) {
        console.error('Error deleting experiment:', error);
        showNotification(`Failed to delete experiment`, 'error');
    }
}

// Quick create experiment with default values
async function quickCreateExperiment() {
    try {
        // Get all existing experiments to generate unique ID
        const response = await apiCall('/experiments');
        const experiments = response.experiments || [];

        // Find next available "new-experiment-N" ID
        let counter = 0;
        let newId = 'new-experiment';

        const existingIds = new Set(experiments.map(exp => exp.id));
        while (existingIds.has(newId)) {
            counter++;
            newId = `new-experiment-${counter}`;
        }

        // Create experiment with minimal required fields
        const request = {
            id: newId,
            name: newId,
            base_command: 'lt',
            tool_path: 'recipes.experiment.cog_arena.train',
            git_branch: null,
            nodes: 1,
            gpus: 0,
            instance_type: null,
            cloud: null,
            spot: false,
            flags: {},
            description: null,
            desired_state: 'STOPPED',
        };

        await apiCall('/experiments', {
            method: 'POST',
            body: JSON.stringify(request),
        });

        showNotification(`Created experiment: ${newId}`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error creating experiment:', error);
        showNotification(error.message || 'Error creating experiment', 'error');
    }
}

// Toggle create experiment form visibility
async function toggleCreateExperiment() {
    const section = document.getElementById('create-section');
    const btn = document.getElementById('new-experiment-btn');

    if (section.style.display === 'none') {
        section.style.display = 'block';
        btn.textContent = '× Cancel';
        btn.classList.remove('btn-primary');
    } else {
        hideCreateExperiment();
    }
}

function hideCreateExperiment() {
    const section = document.getElementById('create-section');
    const btn = document.getElementById('new-experiment-btn');
    const form = document.getElementById('create-form');

    section.style.display = 'none';
    btn.textContent = '+ New Experiment';
    btn.classList.add('btn-primary');
    form.reset();
}

// Create experiment from form
async function createExperiment(event) {
    event.preventDefault();

    try {
        const flagsText = document.getElementById('exp-flags').value;
        let flags = {};

        if (flagsText.trim()) {
            try {
                flags = JSON.parse(flagsText);
            } catch (e) {
                showNotification('Invalid JSON in flags field', 'error');
                return;
            }
        }

        const request = {
            id: document.getElementById('exp-id').value,
            name: document.getElementById('exp-name').value,
            base_command: document.getElementById('exp-command').value,
            tool_path: document.getElementById('exp-tool').value || null,
            git_branch: document.getElementById('exp-git-branch').value || null,
            nodes: parseInt(document.getElementById('exp-nodes').value),
            gpus: parseInt(document.getElementById('exp-gpus').value),
            instance_type: document.getElementById('exp-instance').value || null,
            cloud: document.getElementById('exp-cloud').value || null,
            spot: document.getElementById('exp-spot').checked,
            flags: flags,
            description: document.getElementById('exp-description').value || null,
            desired_state: document.getElementById('exp-start').checked ? 'RUNNING' : 'STOPPED',
        };

        await apiCall('/experiments', {
            method: 'POST',
            body: JSON.stringify(request),
        });

        showNotification(`Created experiment: ${request.id}`, 'success');
        hideCreateExperiment();
        await loadData();
    } catch (error) {
        console.error('Error creating experiment:', error);
        showNotification(error.message || 'Error creating experiment', 'error');
    }
}

// Duplicate experiment
async function duplicateExperiment(experimentId) {
    try {
        const data = await apiCall(`/experiments/${experimentId}`);
        const exp = data.experiment;

        // Create new ID by appending .copy
        let newId = `${exp.id}.copy`;
        let counter = 1;

        // Check if this ID exists, if so increment counter
        while (allExperiments.some(e => e.id === newId)) {
            counter++;
            newId = `${exp.id}.copy${counter}`;
        }

        // Create duplicate with new ID
        const request = {
            id: newId,
            name: exp.name,
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
            group: exp.group,
            order: exp.order,
            desired_state: 'STOPPED',  // Always create duplicates as stopped
        };

        await apiCall('/experiments', {
            method: 'POST',
            body: JSON.stringify(request),
        });

        showNotification(`Created duplicate: ${newId}`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error duplicating experiment:', error);
    }
}

// System actions
async function refreshState() {
    try {
        await apiCall('/refresh', { method: 'POST' });
        showNotification('Refresh complete', 'success');
        await loadData();
    } catch (error) {
        console.error('Error refreshing:', error);
    }
}

async function forceReconcile() {
    try {
        await apiCall('/reconcile', { method: 'POST' });
        showNotification('Reconciliation complete', 'success');
        await loadData();
    } catch (error) {
        console.error('Error reconciling:', error);
    }
}

// Bulk actions
async function bulkStart() {
    const count = selectedExperiments.size;
    if (count === 0) return;

    try {
        const promises = Array.from(selectedExperiments).map(id =>
            apiCall(`/experiments/${id}/state`, {
                method: 'POST',
                body: JSON.stringify({ desired_state: 'RUNNING' }),
            })
        );
        await Promise.all(promises);
        showNotification(`Started ${count} experiments`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error starting experiments:', error);
    }
}

async function bulkStop() {
    const count = selectedExperiments.size;
    if (count === 0) return;

    try {
        const promises = Array.from(selectedExperiments).map(id =>
            apiCall(`/experiments/${id}/state`, {
                method: 'POST',
                body: JSON.stringify({ desired_state: 'STOPPED' }),
            })
        );
        await Promise.all(promises);
        showNotification(`Stopped ${count} experiments`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error stopping experiments:', error);
    }
}

async function bulkDuplicate() {
    const count = selectedExperiments.size;
    if (count === 0) return;

    try {
        for (const experimentId of selectedExperiments) {
            await duplicateExperiment(experimentId);
        }
        showNotification(`Duplicated ${count} experiments`, 'success');
    } catch (error) {
        console.error('Error duplicating experiments:', error);
    }
}

async function bulkDelete() {
    const count = selectedExperiments.size;
    if (count === 0) return;

    // Capture IDs before deleting for undo
    const deletedIds = Array.from(selectedExperiments);

    try {
        const promises = deletedIds.map(id =>
            apiCall(`/experiments/${id}`, { method: 'DELETE' })
        );
        await Promise.all(promises);
        selectedExperiments.clear();

        // Show notification with undo callback
        showNotification(
            `Deleted ${count} experiment${count > 1 ? 's' : ''}`,
            'success',
            async () => {
                // Undo: restore deleted experiments
                try {
                    const restorePromises = deletedIds.map(id =>
                        apiCall(`/experiments/${id}/undelete`, { method: 'POST' })
                    );
                    await Promise.all(restorePromises);
                    showNotification(`Restored ${count} experiment${count > 1 ? 's' : ''}`, 'success');
                    await loadData();
                } catch (error) {
                    console.error('Error restoring experiments:', error);
                    showNotification('Failed to restore experiments', 'error');
                }
            }
        );

        await loadData();
    } catch (error) {
        console.error('Error deleting experiments:', error);
    }
}

// Jobs multiselect and filters
function toggleJobSelection(jobId, checked) {
    if (checked) {
        selectedJobs.add(jobId);
    } else {
        selectedJobs.delete(jobId);
    }
    updateJobsBulkActions();

    const selectAll = document.getElementById('select-all-jobs');
    if (selectAll) {
        // Count how many jobs can be stopped
        const stoppableJobs = allJobs.filter(j =>
            j.status.toLowerCase() === 'running' || j.status.toLowerCase() === 'pending'
        );
        const allSelected = selectedJobs.size === stoppableJobs.length && stoppableJobs.length > 0;
        const someSelected = selectedJobs.size > 0 && selectedJobs.size < stoppableJobs.length;

        selectAll.checked = allSelected;
        selectAll.indeterminate = someSelected;
    }
}

function toggleSelectAllJobs(checked) {
    selectedJobs.clear();
    if (checked) {
        // Only select jobs that can be stopped
        allJobs.forEach(job => {
            if (job.status.toLowerCase() === 'running' || job.status.toLowerCase() === 'pending') {
                selectedJobs.add(job.id);
            }
        });
    }
    const selectAll = document.getElementById('select-all-jobs');
    if (selectAll) {
        selectAll.indeterminate = false;
    }
    updateJobsTable(allJobs);
}

function updateJobsBulkActions() {
    const bulkActions = document.getElementById('jobs-bulk-actions');
    if (bulkActions) {
        bulkActions.style.display = selectedJobs.size > 0 ? 'flex' : 'none';
    }
}

async function bulkStopJobs() {
    const count = selectedJobs.size;
    if (count === 0) return;

    try {
        const promises = Array.from(selectedJobs).map(jobId =>
            apiCall(`/jobs/${jobId}/cancel`, { method: 'POST' })
        );
        await Promise.all(promises);
        selectedJobs.clear();
        showNotification(`Stopped ${count} jobs`, 'success');
        await loadData();
    } catch (error) {
        console.error('Error stopping jobs:', error);
    }
}

// Jobs filter functions
async function toggleStoppedFilter(checked) {
    showStoppedJobs = checked;
    await saveJobsSetting('showStopped', checked);
    await loadData();
}

async function toggleOrphanedFilter(checked) {
    showOrphanedOnly = checked;
    await saveJobsSetting('showOrphaned', checked);
    updateJobsTable(allJobs);
}

async function toggleMyJobsFilter(checked) {
    showMyJobsOnly = checked;
    await saveJobsSetting('showMyJobs', checked);
    updateJobsTable(allJobs);
}

async function updateJobsLimit(value) {
    jobsLimit = parseInt(value) || 20;
    await saveJobsSetting('limit', jobsLimit);
    await loadData();
}

function filterJobs(text) {
    jobsFilterText = text;
    updateJobsTable(allJobs);
}

// Jobs settings persistence
async function loadJobsSettings() {
    try {
        const response = await fetch('/api/settings/jobs-filters');
        if (response.ok) {
            const data = await response.json();
            const settings = data.value || {};

            showStoppedJobs = settings.showStopped || false;
            showOrphanedOnly = settings.showOrphaned || false;
            jobsLimit = settings.limit || 20;
        } else if (response.status === 404) {
            // Settings don't exist yet, use defaults
            console.log('No saved jobs settings found, using defaults');
        }
    } catch (error) {
        console.log('No saved jobs settings found, using defaults');
    }

    // Always update UI to reflect loaded or default values
    const stoppedCheckbox = document.getElementById('filter-stopped');
    if (stoppedCheckbox) stoppedCheckbox.checked = showStoppedJobs;

    const orphanedCheckbox = document.getElementById('filter-orphaned');
    if (orphanedCheckbox) orphanedCheckbox.checked = showOrphanedOnly;

    const limitInput = document.getElementById('jobs-limit');
    if (limitInput) limitInput.value = jobsLimit;
}

async function saveJobsSetting(key, value) {
    try {
        // Get current settings
        let settings = {};
        try {
            const response = await fetch('/api/settings/jobs-filters');
            if (response.ok) {
                const data = await response.json();
                settings = data.value || {};
            }
        } catch (e) {
            // No existing settings
        }

        // Update the specific key
        settings[key] = value;

        // Save back
        await fetch('/api/settings/jobs-filters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });
    } catch (error) {
        console.error('Error saving jobs setting:', error);
    }
}

// Experiment order persistence
async function saveExperimentOrder(order) {
    try {
        await fetch('/api/experiments/reorder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order: order }),
        });
    } catch (error) {
        console.error('Error saving experiment order:', error);
    }
}
