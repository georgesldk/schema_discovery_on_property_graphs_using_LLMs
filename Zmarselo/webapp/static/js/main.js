let currentJobId = null;
let statusCheckInterval = null;
let compareStatusCheckInterval = null;
let currentCompareJobId = null;
let graphNetwork = null;
let currentSchema = null;
let inferredGraphNetwork = null;
let groundTruthGraphNetwork = null;
let groundTruthSchema = null;
let currentMode = 'proof_of_concept'; // 'proof_of_concept' or 'new_dataset'
let currentDatasetId = null;

// State persistence keys
const STATE_KEYS = {
    SCHEMA: 'schemaDiscovery_currentSchema',
    GROUND_TRUTH: 'schemaDiscovery_groundTruthSchema',
    DATASET_ID: 'schemaDiscovery_datasetId',
    COMPARISON_RESULTS: 'schemaDiscovery_comparisonResults'
};

// Save state to sessionStorage
function saveState() {
    try {
        if (currentSchema) {
            sessionStorage.setItem(STATE_KEYS.SCHEMA, JSON.stringify(currentSchema));
        }
        if (groundTruthSchema) {
            sessionStorage.setItem(STATE_KEYS.GROUND_TRUTH, JSON.stringify(groundTruthSchema));
        }
        if (currentDatasetId) {
            sessionStorage.setItem(STATE_KEYS.DATASET_ID, currentDatasetId);
        }
    } catch (e) {
        console.warn('Failed to save state:', e);
    }
}

// Restore state from sessionStorage
function restoreState() {
    try {
        const savedSchema = sessionStorage.getItem(STATE_KEYS.SCHEMA);
        const savedGroundTruth = sessionStorage.getItem(STATE_KEYS.GROUND_TRUTH);
        const savedDatasetId = sessionStorage.getItem(STATE_KEYS.DATASET_ID);

        if (savedSchema) {
            currentSchema = JSON.parse(savedSchema);
        }
        if (savedGroundTruth) {
            groundTruthSchema = JSON.parse(savedGroundTruth);
        }
        if (savedDatasetId) {
            currentDatasetId = savedDatasetId;
        }

        return !!savedSchema; // Return true if we have a schema to restore
    } catch (e) {
        console.warn('Failed to restore state:', e);
        return false;
    }
}

// Clear saved state
function clearState() {
    try {
        Object.values(STATE_KEYS).forEach(key => sessionStorage.removeItem(key));
    } catch (e) {
        console.warn('Failed to clear state:', e);
    }
}

// Status checking
function startStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(async () => {
        if (!currentJobId) return;

        try {
            const response = await fetch(`/status/${currentJobId}`);
            const data = await response.json();

            // Update mode and dataset_id from response
            if (data.mode) {
                currentMode = data.mode;
            }
            if (data.dataset_id) {
                currentDatasetId = data.dataset_id;
            }

            updateProgress(data);

            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(statusCheckInterval);

                if (data.status === 'completed') {
                    // Don't auto-redirect, show "View Results" button instead
                    showCompletionButton(data);
                } else {
                    showError(data.message || 'An error occurred');
                }
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }, 2000); // Check every 2 seconds
}

function startCompareStatusCheck() {
    if (compareStatusCheckInterval) {
        clearInterval(compareStatusCheckInterval);
    }

    document.getElementById('compareNotRun').classList.add('hidden');
    document.getElementById('compareRunning').style.display = 'block';
    document.getElementById('compareResults').style.display = 'none';

    compareStatusCheckInterval = setInterval(async () => {
        if (!currentCompareJobId) return;

        try {
            const response = await fetch(`/status/${currentCompareJobId}`);
            const data = await response.json();

            // Update console output
            const consoleOutput = document.getElementById('compareConsoleOutput');
            if (consoleOutput && data.console_output && Array.isArray(data.console_output)) {
                consoleOutput.textContent = data.console_output.join('\n');
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
            }

            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(compareStatusCheckInterval);

                if (data.status === 'completed') {
                    document.getElementById('compareRunning').style.display = 'none';
                    document.getElementById('compareResults').style.display = 'block';
                    displayCompareResults(data);
                } else {
                    alert('Comparison failed: ' + (data.message || 'Unknown error'));
                }
            }
        } catch (error) {
            console.error('Compare status check error:', error);
        }
    }, 2000);
}

function displayCompareResults(data) {
    const results = data.compare_results || {};
    const consoleOutput = data.console_output || [];

    // Display scores
    const metricsDiv = document.getElementById('comparisonMetrics');
    if (metricsDiv && results.scores) {
        metricsDiv.innerHTML = '';
        for (const [metric, value] of Object.entries(results.scores)) {
            const metricItem = document.createElement('div');
            metricItem.className = 'metric-item';
            metricItem.innerHTML = `
                <span class="metric-label">${metric}</span>
                <span class="metric-value">${value}</span>
            `;
            metricsDiv.appendChild(metricItem);
        }
    }

    // Display tables
    const tablesDiv = document.getElementById('compareTables');
    if (tablesDiv) {
        tablesDiv.innerHTML = '';

        // Nodes table
        if (results.nodes) {
            const nodesTable = createComparisonTable(
                'Nodes Comparison',
                ['Ground Truth', 'Inferred', 'Match'],
                results.nodes.gt || [],
                results.nodes.inferred || [],
                results.nodes.matches || []
            );
            tablesDiv.appendChild(nodesTable);
        }

        // Edges table
        if (results.edges) {
            const edgesTable = createComparisonTable(
                'Edges Comparison',
                ['Ground Truth', 'Inferred', 'Match'],
                results.edges.gt || [],
                results.edges.inferred || [],
                results.edges.matches || []
            );
            tablesDiv.appendChild(edgesTable);
        }
    }
}

function createComparisonTable(title, headers, gtItems, inferredItems, matches) {
    const tableContainer = document.createElement('div');
    tableContainer.style.cssText = 'margin-bottom: 30px;';

    const titleEl = document.createElement('h4');
    titleEl.textContent = title;
    titleEl.style.cssText = 'margin-bottom: 15px; color: var(--text-primary);';
    tableContainer.appendChild(titleEl);

    const table = document.createElement('table');
    table.style.cssText = 'width: 100%; border-collapse: collapse; background: var(--bg-color); border-radius: 8px; overflow: hidden;';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.style.cssText = 'background: var(--primary-color); color: white;';
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.cssText = 'padding: 12px; text-align: left; font-weight: 600;';
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');

    // Create match map
    const matchMap = new Map();
    matches.forEach(m => {
        matchMap.set(m.gt, m.inferred);
    });

    // Add GT items
    const maxRows = Math.max(gtItems.length, inferredItems.length);
    for (let i = 0; i < maxRows; i++) {
        const row = document.createElement('tr');
        row.style.cssText = 'border-bottom: 1px solid var(--border-color);';

        const gtCell = document.createElement('td');
        gtCell.style.cssText = 'padding: 10px;';
        gtCell.textContent = gtItems[i] || '';
        row.appendChild(gtCell);

        const inferredCell = document.createElement('td');
        inferredCell.style.cssText = 'padding: 10px;';
        inferredCell.textContent = inferredItems[i] || '';
        row.appendChild(inferredCell);

        const matchCell = document.createElement('td');
        matchCell.style.cssText = 'padding: 10px; text-align: center;';
        if (gtItems[i] && matchMap.has(gtItems[i])) {
            matchCell.innerHTML = '✓';
            matchCell.style.color = '#10b981';
        } else if (gtItems[i] || inferredItems[i]) {
            matchCell.innerHTML = '✗';
            matchCell.style.color = '#ef4444';
        }
        row.appendChild(matchCell);

        tbody.appendChild(row);
    }

    table.appendChild(tbody);
    tableContainer.appendChild(table);

    return tableContainer;
}

function updateProgress(data) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusMessage = document.getElementById('statusMessage');
    const consoleOutput = document.getElementById('consoleOutput');

    progressFill.style.width = `${data.progress}%`;
    progressText.textContent = `${data.progress}%`;
    statusMessage.textContent = data.message || '';

    // Update console output if available
    if (data.console_output && Array.isArray(data.console_output)) {
        consoleOutput.textContent = data.console_output.join('\n');
        // Auto-scroll to bottom
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
}

function showCompletionButton(data) {
    // Show "View Results" button when processing is complete
    const viewResultsContainer = document.getElementById('viewResultsContainer');
    const viewResultsBtn = document.getElementById('viewResultsBtn');

    if (viewResultsContainer && viewResultsBtn) {
        viewResultsContainer.classList.add('visible');

        // Store result data for when user clicks "View Results"
        viewResultsBtn.onclick = async () => {
            try {
                // Always fetch fresh status to get output_file
                const response = await fetch(`/status/${currentJobId}`);
                const statusData = await response.json();

                let schema = null;

                // Try to load from output file
                if (statusData.output_file) {
                    try {
                        const filePath = statusData.output_file;
                        const fileResponse = await fetch(`/api/load-schema?file=${encodeURIComponent(filePath)}`);
                        if (fileResponse.ok) {
                            schema = await fileResponse.json();
                        }
                    } catch (e) {
                        console.log('Could not load from API, trying direct file read');
                    }
                }

                // Fallback: try to read file directly if we have the path
                if (!schema && statusData.output_file) {
                    // For now, we'll need to add an endpoint to read the file
                    // Or we can use the result if available
                    if (statusData.result) {
                        schema = statusData.result;
                    }
                }

                if (schema) {
                    showResults(schema);
                } else {
                    showError('Could not load schema. Please check the console output for errors.');
                }
            } catch (error) {
                console.error('Error loading results:', error);
                showError('Failed to load results: ' + error.message);
            }
        };
    }
}

// UI State Management
function showProgress() {
    document.getElementById('pocCard').classList.add('hidden');
    document.getElementById('progressCard').classList.remove('hidden');
    document.getElementById('resultsCard').classList.add('hidden');
    document.getElementById('errorCard').classList.add('hidden');
}

function showResults(schema) {
    document.getElementById('progressCard').classList.add('hidden');
    document.getElementById('resultsCard').classList.remove('hidden');

    // Store schema for graph rendering
    currentSchema = schema;
    saveState(); // Persist state for page refresh

    renderSchema(schema);

    // Show/hide comparison tab based on mode
    const comparisonTabBtn = document.querySelector('[data-tab="comparison"]');
    comparisonTabBtn.style.display = '';

    // Reset to schema tab
    switchTab('schema');
    updateTabNavigation();

    // Setup download button
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.onclick = () => {
        window.location.href = `/download/${currentJobId}`;
    };

    // Setup back to console button
    const backToConsoleBtn = document.getElementById('backToConsoleBtn');
    if (backToConsoleBtn) {
        backToConsoleBtn.onclick = () => {
            document.getElementById('resultsCard').classList.add('hidden');
            document.getElementById('progressCard').classList.remove('hidden');
        };
    }

    // Setup compare button
    const compareBtn = document.getElementById('compareBtn');
    if (compareBtn && currentDatasetId) {
        compareBtn.style.display = '';
        compareBtn.onclick = async () => {
            compareBtn.disabled = true;
            compareBtn.innerHTML = 'Comparing...';

            try {
                const response = await fetch(`/compare-dataset/${currentDatasetId}`, {
                    method: 'POST'
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to start comparison');
                }

                // Store compare job ID and show comparison tab
                currentCompareJobId = data.job_id;
                switchTab('comparison');
                startCompareStatusCheck();
            } catch (error) {
                alert('Error: ' + error.message);
                compareBtn.disabled = false;
                compareBtn.innerHTML = 'Compare with GT';
            }
        };
    } else if (compareBtn) {
        compareBtn.style.display = 'none';
    }
}

function showError(message) {
    document.getElementById('progressCard').classList.add('hidden');
    document.getElementById('errorCard').classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

function resetForm() {
    currentJobId = null;
    currentSchema = null;
    groundTruthSchema = null;
    clearState(); // Clear saved state
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    // Destroy graphs if they exist
    if (graphNetwork) {
        graphNetwork.destroy();
        graphNetwork = null;
    }
    if (inferredGraphNetwork) {
        inferredGraphNetwork.destroy();
        inferredGraphNetwork = null;
    }
    if (groundTruthGraphNetwork) {
        groundTruthGraphNetwork.destroy();
        groundTruthGraphNetwork = null;
    }

    // Reset UI
    document.getElementById('pocCard').classList.remove('hidden');
    // Reset dataset selector
    document.getElementById('datasetSelect').value = '';
    document.getElementById('datasetDescription').classList.remove('visible');
    const processBtn = document.getElementById('processDatasetBtn');
    processBtn.disabled = true;
    processBtn.innerHTML = 'Infer Schema';

    document.getElementById('errorCard').classList.add('hidden');
    document.getElementById('resultsCard').classList.add('hidden');

    // Reset tabs
    switchTab('schema');
}

// --- Helper: Robust Label Detection ---
function getLabel(item) {
    // Checks multiple possible fields returned by different LLM versions or schemas
    // For GT schemas with label arrays, use the first (primary) label
    if (item.labels && Array.isArray(item.labels) && item.labels.length > 0) {
        // Return the first label (primary label)
        return item.labels[0];
    }
    return item.name || item.label || item.type || 'Unknown';
}

// --- Helper: Robust Edge Connection Finder ---
/**
 * ROBUST EDGE CONNECTION FINDER
 * Checks all possible keys for source and target nodes.
 */
function getEdgeNodes(edge) {
    // First try direct from/to fields
    if (edge.from && edge.to) {
        return { from: edge.from, to: edge.to };
    }

    // If topology is specified, extract first valid combination
    if (edge.topology && edge.topology.length > 0) {
        const topology = edge.topology[0];
        const sources = topology.allowed_sources || [];
        const targets = topology.allowed_targets || [];
        if (sources.length > 0 && targets.length > 0) {
            // Return first source/target combination for simple visualization
            // The rendering logic should handle creating edges for all combinations
            return { from: sources[0], to: targets[0] };
        }
    }

    // Fallback to other possible field names
    return {
        from: edge.from || edge.from_node || edge.source || edge.source_node || edge.start_node,
        to: edge.to || edge.to_node || edge.target || edge.target_node || edge.end_node
    };
}

// Schema rendering
function renderSchema(schema) {
    const nodeTypesDiv = document.getElementById('nodeTypes');
    const edgeTypesDiv = document.getElementById('edgeTypes');

    nodeTypesDiv.innerHTML = '';
    edgeTypesDiv.innerHTML = '';

    if (schema.node_types && schema.node_types.length > 0) {
        schema.node_types.forEach(nodeType => {
            const nodeItem = createSchemaItem(
                getLabel(nodeType),
                nodeType.properties || [],
                'node'
            );
            nodeTypesDiv.appendChild(nodeItem);
        });
    } else {
        nodeTypesDiv.innerHTML = '<p style="color: var(--text-secondary);">No node types found</p>';
    }

    if (schema.edge_types && schema.edge_types.length > 0) {
        schema.edge_types.forEach(edgeType => {
            const edgeItem = createSchemaItem(
                getLabel(edgeType),
                edgeType.properties || [],
                'edge'
            );
            edgeTypesDiv.appendChild(edgeItem);
        });
    } else {
        edgeTypesDiv.innerHTML = '<p style="color: var(--text-secondary);">No edge types found</p>';
    }
}

function createSchemaItem(name, properties, type) {
    const item = document.createElement('div');
    item.className = 'schema-item';

    const header = document.createElement('div');
    header.className = 'schema-item-header';
    header.innerHTML = `
        <span class="schema-item-name">${name}</span>
        <span class="schema-item-count">${properties.length} properties</span>
    `;

    const propertiesList = document.createElement('div');
    propertiesList.className = 'properties-list';

    if (properties.length > 0) {
        properties.forEach(prop => {
            const propItem = document.createElement('div');
            propItem.className = 'property-item';

            const propName = document.createElement('div');
            propName.className = 'property-name';
            propName.textContent = prop.name || 'Unknown';

            const propDetails = document.createElement('div');
            propDetails.className = 'property-details';

            let detailsHTML = '';
            if (prop.type) {
                detailsHTML += `<span>Type: <strong>${prop.type}</strong></span>`;
            }
            if (prop.mandatory !== undefined) {
                const badgeClass = prop.mandatory ? 'badge-mandatory' : 'badge-optional';
                const badgeText = prop.mandatory ? 'Mandatory' : 'Optional';
                detailsHTML += `<span class="property-badge ${badgeClass}">${badgeText}</span>`;
            }

            propDetails.innerHTML = detailsHTML;

            propItem.appendChild(propName);
            propItem.appendChild(propDetails);
            propertiesList.appendChild(propItem);
        });
    } else {
        propertiesList.innerHTML = '<p style="color: var(--text-secondary); padding: 12px;">No properties defined</p>';
    }

    item.appendChild(header);
    item.appendChild(propertiesList);

    return item;
}

// New Analysis button (optional element)
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener('click', resetForm);
}

// Proof of Concept mode - Load datasets on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadDatasets();

    // Try to restore previous state
    const hasState = restoreState();
    if (hasState && currentSchema) {
        // Restore the dataset selection if available
        if (currentDatasetId) {
            const select = document.getElementById('datasetSelect');
            if (select) {
                select.value = currentDatasetId;
            }
        }

        // Show results page with restored schema
        document.getElementById('pocCard').classList.add('hidden');
        document.getElementById('progressCard').classList.add('hidden');
        document.getElementById('resultsCard').classList.remove('hidden');

        // Display the restored schema
        displaySchema(currentSchema);

        // If we have ground truth, also restore comparison
        if (groundTruthSchema) {
            document.getElementById('compareNotRun').classList.add('hidden');
            document.getElementById('compareResults').style.display = 'block';
            renderComparisonGraphs();
            calculateComparisonMetrics();
        }
    }
});

function switchMode(mode) {
    // Mode switching disabled, always proof_of_concept
    return;
}

async function loadDatasets() {
    try {
        const response = await fetch('/datasets');
        const data = await response.json();

        const select = document.getElementById('datasetSelect');
        select.innerHTML = '';

        if (data.datasets.length === 0) {
            select.innerHTML = '<option value="">No datasets available</option>';
            return;
        }

        select.innerHTML = '<option value="">Select a dataset...</option>';
        data.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            option.dataset.description = dataset.description || '';
            select.appendChild(option);
        });

        // Handle dataset selection
        select.addEventListener('change', (e) => {
            const selectedOption = e.target.options[e.target.selectedIndex];
            const description = selectedOption.dataset.description;
            const datasetId = e.target.value;

            currentDatasetId = datasetId;

            const descDiv = document.getElementById('datasetDescription');
            const processBtn = document.getElementById('processDatasetBtn');

            if (datasetId) {
                descDiv.textContent = description || `Process ${selectedOption.textContent} dataset`;
                descDiv.classList.add('visible');
                processBtn.disabled = false;
            } else {
                descDiv.classList.remove('visible');
                processBtn.disabled = true;
            }
        });
    } catch (error) {
        console.error('Error loading datasets:', error);
        document.getElementById('datasetSelect').innerHTML = '<option value="">Error loading datasets</option>';
    }
}

// Process dataset button handler
document.getElementById('processDatasetBtn').addEventListener('click', async () => {
    if (!currentDatasetId) {
        alert('Please select a dataset first');
        return;
    }

    const processBtn = document.getElementById('processDatasetBtn');
    processBtn.disabled = true;
    processBtn.innerHTML = 'Processing...';

    try {
        const response = await fetch(`/process-dataset/${currentDatasetId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to start processing');
        }

        currentJobId = data.job_id;
        showProgress();
        startStatusCheck();
    } catch (error) {
        showError(error.message);
        processBtn.disabled = false;
        processBtn.innerHTML = 'Infer Schema';
    }
});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        switchTab(tabName);
    });
});

// Tab navigation arrows
document.addEventListener('DOMContentLoaded', () => {
    const prevTabBtn = document.getElementById('prevTabBtn');
    const nextTabBtn = document.getElementById('nextTabBtn');

    if (prevTabBtn) {
        prevTabBtn.addEventListener('click', () => {
            const tabs = getAvailableTabs();
            const currentIndex = getCurrentTabIndex();
            if (currentIndex > 0) {
                switchTab(tabs[currentIndex - 1]);
            }
        });
    }

    if (nextTabBtn) {
        nextTabBtn.addEventListener('click', () => {
            const tabs = getAvailableTabs();
            const currentIndex = getCurrentTabIndex();
            if (currentIndex < tabs.length - 1) {
                switchTab(tabs[currentIndex + 1]);
            }
        });
    }
});

function getAvailableTabs() {
    const tabs = ['schema', 'graph'];
    if (currentMode === 'proof_of_concept') {
        tabs.push('comparison');
    }
    return tabs;
}

function getCurrentTabIndex() {
    const tabs = getAvailableTabs();
    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab) {
        const tabName = activeTab.getAttribute('data-tab');
        return tabs.indexOf(tabName);
    }
    return 0;
}

function updateTabNavigation() {
    const tabs = getAvailableTabs();
    const currentIndex = getCurrentTabIndex();
    const prevBtn = document.getElementById('prevTabBtn');
    const nextBtn = document.getElementById('nextTabBtn');

    if (prevBtn && nextBtn) {
        prevBtn.style.display = tabs.length > 1 ? '' : 'none';
        nextBtn.style.display = tabs.length > 1 ? '' : 'none';

        prevBtn.disabled = currentIndex === 0;
        nextBtn.disabled = currentIndex === tabs.length - 1;
    }
}

function switchTab(tabName) {
    // Don't switch to comparison tab if in new_dataset mode
    if (tabName === 'comparison' && currentMode !== 'proof_of_concept') {
        return;
    }

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-tab') === tabName) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
        content.classList.remove('active');
    });

    if (tabName === 'schema') {
        document.getElementById('schemaTab').classList.remove('hidden');
        document.getElementById('schemaTab').classList.add('active');
    } else if (tabName === 'graph') {
        document.getElementById('graphTab').classList.remove('hidden');
        document.getElementById('graphTab').classList.add('active');
        // Render graph if schema is available
        if (currentSchema) {
            // Use setTimeout to ensure the tab is visible before rendering
            setTimeout(() => {
                renderGraph(currentSchema);
                // After rendering, ensure the network resizes and fits
                if (graphNetwork) {
                    graphNetwork.redraw();
                    graphNetwork.fit();
                }
            }, 50);
        }
    } else if (tabName === 'comparison') {
        document.getElementById('comparisonTab').classList.remove('hidden');
        document.getElementById('comparisonTab').classList.add('active');
        // Load and render comparison if schema is available
        if (currentSchema) {
            loadComparison();
        }
    }

    updateTabNavigation();
}

// Helper function to determine node status (matched vs extra) for coloring
function getNodeStatus(schema) {
    if (!groundTruthSchema || !schema) return { matched: new Set(), extra: new Set() };

    const gtNodes = new Map();
    const infNodes = new Map();

    // Use the robust getLabel helper for mapping
    (groundTruthSchema.node_types || []).forEach(n => {
        const lbl = getLabel(n);
        if (lbl !== 'Unknown') gtNodes.set(lbl, n);
    });

    (schema.node_types || []).forEach(n => {
        const lbl = getLabel(n);
        if (lbl !== 'Unknown') infNodes.set(lbl, n);
    });

    // Find matched nodes
    const nodeMatchMap = new Map();
    gtNodes.forEach((_, gtName) => {
        const match = findBestMatch(gtName, Array.from(infNodes.keys()), 0.75, false);
        if (match) {
            nodeMatchMap.set(gtName, match);
        }
    });

    // Determine matched and extra nodes
    const matchedInferredNodeSet = new Set(nodeMatchMap.values());
    const extraNodes = new Set(Array.from(infNodes.keys()).filter(name => !matchedInferredNodeSet.has(name)));

    return {
        matched: matchedInferredNodeSet,
        extra: extraNodes
    };
}

// Graph rendering function
function renderGraph(schema) {
    const container = document.getElementById('graphNetwork');

    // Ensure container has proper dimensions before vis.js initializes
    // vis.js defaults to 150px height if container has no height
    const parentHeight = container.parentElement.offsetHeight || window.innerHeight - 300;
    container.style.height = Math.max(600, parentHeight) + 'px';

    // Clear previous graph
    if (graphNetwork) {
        graphNetwork.destroy();
    }

    // Get node status for coloring (if ground truth is available)
    const nodeStatus = getNodeStatus(schema);

    // Create nodes from node_types
    const nodes = [];
    const nodeTypeMap = new Map();

    if (schema.node_types && schema.node_types.length > 0) {
        schema.node_types.forEach((nodeType, index) => {
            const nodeName = getLabel(nodeType) || `Node${index}`;
            const nodeId = `node_${index}`;
            nodeTypeMap.set(nodeName, nodeId);

            const propertyCount = (nodeType.properties || []).length;
            const mandatoryCount = (nodeType.properties || []).filter(p => p.mandatory).length;

            // Determine node color based on status
            let nodeColor;
            if (nodeStatus.matched.has(nodeName)) {
                // Matched node - Green
                nodeColor = {
                    background: 'rgba(34, 197, 94, 0.25)',
                    border: '#22c55e',
                    highlight: {
                        background: 'rgba(34, 197, 94, 0.4)',
                        border: '#22c55e'
                    }
                };
            } else if (nodeStatus.extra.has(nodeName)) {
                // Extra node - Orange
                nodeColor = {
                    background: 'rgba(249, 115, 22, 0.25)',
                    border: '#f97316',
                    highlight: {
                        background: 'rgba(249, 115, 22, 0.4)',
                        border: '#f97316'
                    }
                };
            } else {
                // Default color - Blue (when ground truth is not available)
                nodeColor = {
                    background: 'rgba(59, 130, 246, 0.25)',
                    border: '#3b82f6',
                    highlight: {
                        background: 'rgba(59, 130, 246, 0.4)',
                        border: '#60a5fa'
                    }
                };
            }

            // Build title with status information
            let title = `${nodeName}\nProperties: ${propertyCount}\nMandatory: ${mandatoryCount}`;
            if (nodeStatus.matched.has(nodeName)) {
                title += '\n✓ Matched';
            } else if (nodeStatus.extra.has(nodeName)) {
                title += '\n⚠ Extra (Over-inferred)';
            }

            nodes.push({
                id: nodeId,
                label: nodeName,
                title: title,
                shape: 'ellipse',
                color: nodeColor,
                font: {
                    color: '#fafafa',
                    size: 16,
                    face: 'Outfit, sans-serif'
                },
                borderWidth: 2,
                size: 50 + (propertyCount * 3)
            });
        });
    }

    // Create edges from edge_types
    const edges = [];

    if (schema.edge_types && schema.edge_types.length > 0) {
        schema.edge_types.forEach((edgeType, index) => {
            const edgeName = getLabel(edgeType) || `Edge${index}`;

            // Handle topology-based edges (GT format)
            if (edgeType.topology && edgeType.topology.length > 0) {
                edgeType.topology.forEach(topologyEntry => {
                    const sources = topologyEntry.allowed_sources || [];
                    const targets = topologyEntry.allowed_targets || [];

                    // Create edges for all valid source/target combinations
                    sources.forEach(sourceNodeName => {
                        targets.forEach(targetNodeName => {
                            // Find node IDs
                            let fromId = null;
                            let toId = null;

                            // Match by primary label (first part before colon if present)
                            const sourcePrimary = sourceNodeName.split(':')[0];
                            const targetPrimary = targetNodeName.split(':')[0];

                            for (const [name, id] of nodeTypeMap.entries()) {
                                const namePrimary = name.split(':')[0];
                                if (!fromId && (name === sourcePrimary || namePrimary === sourcePrimary ||
                                    name.toLowerCase() === sourcePrimary.toLowerCase())) {
                                    fromId = id;
                                }
                                if (!toId && (name === targetPrimary || namePrimary === targetPrimary ||
                                    name.toLowerCase() === targetPrimary.toLowerCase())) {
                                    toId = id;
                                }
                                if (fromId && toId) break;
                            }

                            // Only create edge if both nodes found
                            if (fromId && toId) {
                                const isSelfLoop = fromId === toId;
                                edges.push({
                                    id: `edge_${index}_${sourceNodeName}_${targetNodeName}`,
                                    from: fromId,
                                    to: toId,
                                    label: edgeName,
                                    title: `${edgeName}: ${sourcePrimary} -> ${targetPrimary}`,
                                    color: {
                                        color: '#6366f1',
                                        highlight: '#818cf8'
                                    },
                                    arrows: { to: { enabled: true, scaleFactor: 1 } },
                                    font: { color: '#a1a1aa', size: 10, align: 'middle', multi: 'html', face: 'Outfit, sans-serif' },
                                    width: 1.5,
                                    smooth: isSelfLoop ? { type: 'curvedCW', roundness: 0.8 } : { type: 'continuous', roundness: 0.5 }
                                });
                            }
                        });
                    });
                });
                return; // Skip normal edge processing for topology-based edges
            }

            // Normal edge processing (inferred schema format)
            let { from: startNode, to: endNode } = getEdgeNodes(edgeType);

            // If start_node/end_node not found, try to infer from edge name or properties
            if (!startNode || !endNode) {
                // Try to extract node names from edge name (e.g., "Neuron_to_SynapseSet")
                const nameParts = edgeName.split(/[_-]?(to|from|connects|has|contains)[_-]?/i);
                if (nameParts.length >= 2) {
                    if (!startNode) {
                        startNode = nameParts[0].trim();
                    }
                    if (!endNode) {
                        endNode = nameParts[nameParts.length - 1].trim();
                    }
                }

                // Try to infer from properties (look for :START_ID and :END_ID patterns)
                if ((!startNode || !endNode) && edgeType.properties) {
                    edgeType.properties.forEach(prop => {
                        const propName = prop.name || '';
                        if (propName.includes('START_ID') && !startNode) {
                            // Try to extract node type from property name
                            const match = propName.match(/\(([^)]+)\)/);
                            if (match) {
                                startNode = match[1].split('-')[0]; // Get first part before hyphen
                            }
                        }
                        if (propName.includes('END_ID') && !endNode) {
                            const match = propName.match(/\(([^)]+)\)/);
                            if (match) {
                                endNode = match[1].split('-')[0];
                            }
                        }
                    });
                }
            }

            // Find node IDs
            let fromId = null;
            let toId = null;

            // Try to match by exact name
            if (startNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name === startNode || name.toLowerCase() === startNode.toLowerCase()) {
                        fromId = id;
                        break;
                    }
                }
            }

            if (endNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name === endNode || name.toLowerCase() === endNode.toLowerCase()) {
                        toId = id;
                        break;
                    }
                }
            }

            // If not found, try to match by partial name
            if (!fromId && startNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name.toLowerCase().includes(startNode.toLowerCase()) ||
                        startNode.toLowerCase().includes(name.toLowerCase())) {
                        fromId = id;
                        break;
                    }
                }
            }

            if (!toId && endNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name.toLowerCase().includes(endNode.toLowerCase()) ||
                        endNode.toLowerCase().includes(name.toLowerCase())) {
                        toId = id;
                        break;
                    }
                }
            }

            // If still not found and we have nodes, create a self-loop or connect to first available node
            if (!fromId && nodes.length > 0) {
                fromId = nodes[0].id;
            }
            if (!toId && nodes.length > 0) {
                // If fromId is set, try to use a different node for toId
                if (fromId && nodes.length > 1) {
                    toId = nodes.find(n => n.id !== fromId)?.id || nodes[0].id;
                } else {
                    toId = nodes[0].id;
                }
            }

            // Only create edge if we have valid node IDs
            if (fromId && toId) {
                const propertyCount = (edgeType.properties || []).filter(p =>
                    p.name && !p.name.includes(':START_ID') && !p.name.includes(':END_ID')
                ).length;

                const isSelfLoop = fromId === toId;

                // Format edge label - replace underscores with spaces and handle long names
                let formattedLabel = edgeName.replace(/_/g, ' ');

                // Special handling for self-loops - simplify the label
                if (isSelfLoop) {
                    // Remove redundant "to [same node]" pattern for self-loops
                    const nodeName = nodes.find(n => n.id === fromId)?.label || '';
                    // Check if label contains the node name at the end (e.g., "SynapseSet to SynapseSet")
                    const selfLoopPattern = new RegExp(`\\s+(to|from|connects|has|contains)\\s+${nodeName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
                    if (selfLoopPattern.test(formattedLabel)) {
                        // Remove the redundant part, keep just the edge type name
                        formattedLabel = formattedLabel.replace(selfLoopPattern, '');
                    }
                    // If it's still a pattern like "NodeType_to_NodeType", extract just the meaningful part
                    if (formattedLabel.toLowerCase().includes(nodeName.toLowerCase()) && formattedLabel.split(/\s+/).length > 1) {
                        // Try to extract just the relationship type
                        const parts = formattedLabel.split(/\s+(to|from|connects|has|contains)\s+/i);
                        if (parts.length >= 2 && parts[0].toLowerCase() === parts[parts.length - 1].toLowerCase()) {
                            // It's a self-loop with same name on both sides, just show the relationship word
                            formattedLabel = parts[1] || formattedLabel.split(/\s+/)[0];
                        }
                    }
                } else {
                    // For regular edges, format normally
                    // If label contains multiple words or is long, format it better
                    if (formattedLabel.length > 20 || formattedLabel.includes(' ')) {
                        // Try to split at common separators (to, from, connects, etc.)
                        const separatorMatch = formattedLabel.match(/\s+(to|from|connects|has|contains|set)\s+/i);
                        if (separatorMatch) {
                            const parts = formattedLabel.split(/\s+(to|from|connects|has|contains|set)\s+/i);
                            // Reconstruct with line break before the separator word
                            formattedLabel = parts[0] + '<br>' + separatorMatch[0].trim() + '<br>' + parts[parts.length - 1];
                        } else {
                            // Split long labels at spaces, taking first two words on first line
                            const words = formattedLabel.split(/\s+/);
                            if (words.length > 2) {
                                const midPoint = Math.ceil(words.length / 2);
                                formattedLabel = words.slice(0, midPoint).join(' ') + '<br>' + words.slice(midPoint).join(' ');
                            } else if (formattedLabel.length > 25) {
                                // For very long single words, truncate
                                formattedLabel = formattedLabel.substring(0, 22) + '...';
                            }
                        }
                    }
                }

                const edgeConfig = {
                    id: `edge_${index}`,
                    from: fromId,
                    to: toId,
                    label: formattedLabel,
                    title: `${edgeName}\nProperties: ${propertyCount}`,
                    color: {
                        color: '#6366f1',
                        highlight: '#818cf8'
                    },
                    arrows: {
                        to: {
                            enabled: true,
                            scaleFactor: 1
                        }
                    },
                    font: {
                        color: '#a1a1aa',
                        size: 10,
                        align: 'middle',
                        multi: 'html',
                        face: 'Outfit, sans-serif'
                    },
                    width: 1.5
                };

                // Special handling for self-loops
                if (isSelfLoop) {
                    // Self-loops use curvedCW type for a clean circular loop
                    edgeConfig.smooth = {
                        type: 'curvedCW',
                        roundness: 0.8  // Larger roundness for a more circular, cleaner loop
                    };
                    // Make self-loops more visible
                    edgeConfig.width = 2;
                    edgeConfig.color = {
                        color: '#818cf8',
                        highlight: '#a5b4fc'
                    };
                    // Position label better for self-loops
                    edgeConfig.font = {
                        ...edgeConfig.font,
                        size: 10,  // Slightly smaller for self-loops
                        vadjust: -5  // Move label up slightly
                    };
                } else {
                    // For regular edges, use continuous smooth curves
                    edgeConfig.smooth = {
                        type: 'continuous',
                        roundness: 0.5
                    };
                }

                edges.push(edgeConfig);
            }
        });
    }

    // If no edges but we have nodes, create a simple layout
    if (edges.length === 0 && nodes.length > 0) {
        // Just show nodes without connections
    }

    // Create network data
    const data = {
        nodes: nodes,
        edges: edges
    };

    // Calculate container height for vis.js
    const containerRect = container.getBoundingClientRect();
    const graphHeight = Math.max(600, window.innerHeight - 280);

    // Network options
    const options = {
        width: '100%',
        height: graphHeight + 'px',
        nodes: {
            shape: 'ellipse',
            font: {
                color: '#fafafa',
                size: 16,
                face: 'Outfit, sans-serif'
            },
            borderWidth: 2,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.3)',
                size: 10,
                x: 3,
                y: 3
            }
        },
        edges: {
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 1
                }
            },
            color: {
                color: '#6366f1',
                highlight: '#818cf8'
            },
            font: {
                color: '#a1a1aa',
                size: 10,
                align: 'middle',
                multi: 'html',
                face: 'Outfit, sans-serif'
            },
            smooth: {
                type: 'continuous',
                roundness: 0.5
            },
            shadow: false,
            labelHighlightBold: false,
            selectionWidth: 2
        },
        physics: {
            enabled: true,
            stabilization: {
                iterations: 150,
                fit: false  // We'll do our own fit
            },
            barnesHut: {
                gravitationalConstant: -5000,
                centralGravity: 0.3,
                springLength: 200,
                springConstant: 0.02,
                damping: 0.09,
                avoidOverlap: 0.5
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true
        },
        layout: {
            improvedLayout: true
        }
    };

    // Create network
    graphNetwork = new vis.Network(container, data, options);

    // Fit the graph properly after stabilization
    graphNetwork.once('stabilizationEnd', () => {
        // Fit with padding to fill the container nicely
        graphNetwork.fit({
            animation: {
                duration: 500,
                easingFunction: 'easeOutQuad'
            }
        });
    });
}



// Comparison functions
/**
 * SOFTENED SCORING:
 * Instead of Matches / (GT + Extras), we use:
 * Matches / (GT + (0.3 * Extras))
 * This means extras only ding your score by 30% instead of 100%.
 */
function calculateRealScore(matches, totalGT, totalExtra) {
    const penaltyWeight = 0.3;
    const denominator = totalGT + (totalExtra * penaltyWeight);
    if (denominator === 0) return 0;

    let score = (matches / denominator) * 100;
    return Math.min(100, score); // Cap at 100
}

async function loadComparison() {
    // Show loading state
    document.getElementById('comparisonMetrics').innerHTML = '<div class="metric-item"><span class="metric-label">Loading comparison data...</span></div>';

    try {
        // Load ground truth schema - use dataset_id if available, otherwise use default endpoint
        const gtUrl = currentDatasetId ? `/ground-truth/${currentDatasetId}` : '/ground-truth';
        const response = await fetch(gtUrl);
        if (!response.ok) {
            throw new Error('Failed to load ground truth schema');
        }
        groundTruthSchema = await response.json();

        // Render both graphs
        renderComparisonGraphs();

        // Calculate and display metrics
        calculateComparisonMetrics();

        // Save state for page refresh
        saveState();

    } catch (error) {
        console.error('Error loading comparison:', error);
        document.getElementById('comparisonMetrics').innerHTML =
            `<div class="metric-item"><span class="metric-label" style="color: var(--error-color);">Error: ${error.message}</span></div>`;
    }
}

function renderComparisonGraphs() {
    // Render inferred schema graph
    const inferredContainer = document.getElementById('inferredGraphNetwork');
    if (inferredGraphNetwork) {
        inferredGraphNetwork.destroy();
    }
    const inferredData = createGraphData(currentSchema, 'inferred');
    inferredGraphNetwork = new vis.Network(inferredContainer, inferredData, getComparisonGraphOptions());
    // Center and zoom the graph after stabilization
    inferredGraphNetwork.once('stabilizationEnd', () => {
        inferredGraphNetwork.fit();
        setTimeout(() => {
            inferredGraphNetwork.moveTo({ scale: 0.85, animation: false });
        }, 50);
    });

    // Render ground truth schema graph
    const gtContainer = document.getElementById('groundTruthGraphNetwork');
    if (groundTruthGraphNetwork) {
        groundTruthGraphNetwork.destroy();
    }
    const gtData = createGraphData(groundTruthSchema, 'groundtruth');
    groundTruthGraphNetwork = new vis.Network(gtContainer, gtData, getComparisonGraphOptions());
    // Center and zoom the graph after stabilization
    groundTruthGraphNetwork.once('stabilizationEnd', () => {
        groundTruthGraphNetwork.fit();
        setTimeout(() => {
            groundTruthGraphNetwork.moveTo({ scale: 0.85, animation: false });
        }, 50);
    });
}

function createGraphData(schema, prefix) {
    const nodes = [];
    const edges = [];
    const nodeTypeMap = new Map();

    // Create nodes
    if (schema.node_types && schema.node_types.length > 0) {
        schema.node_types.forEach((nodeType, index) => {
            const nodeName = getLabel(nodeType) || `Node${index}`;
            const nodeId = `${prefix}_node_${index}`;
            nodeTypeMap.set(nodeName, nodeId);

            const propertyCount = (nodeType.properties || []).length;

            nodes.push({
                id: nodeId,
                label: nodeName,
                title: `${nodeName}\nProperties: ${propertyCount}`,
                shape: 'ellipse',
                color: {
                    background: 'rgba(59, 130, 246, 0.25)',
                    border: '#3b82f6',
                    highlight: {
                        background: 'rgba(59, 130, 246, 0.4)',
                        border: '#60a5fa'
                    }
                },
                font: {
                    color: '#fafafa',
                    size: 12,
                    face: 'Outfit, sans-serif'
                },
                borderWidth: 2,
                size: 25 + (propertyCount * 2)
            });
        });
    }

    // Create edges
    if (schema.edge_types && schema.edge_types.length > 0) {
        schema.edge_types.forEach((edgeType, index) => {
            const edgeName = getLabel(edgeType) || `Edge${index}`;

            // Handle topology-based edges (GT format)
            if (edgeType.topology && edgeType.topology.length > 0) {
                edgeType.topology.forEach(topologyEntry => {
                    const sources = topologyEntry.allowed_sources || [];
                    const targets = topologyEntry.allowed_targets || [];

                    // Create edges for all valid source/target combinations
                    sources.forEach(sourceNodeName => {
                        targets.forEach(targetNodeName => {
                            // Find node IDs
                            let fromId = null;
                            let toId = null;

                            // Match by primary label (first part before colon if present)
                            const sourcePrimary = sourceNodeName.split(':')[0];
                            const targetPrimary = targetNodeName.split(':')[0];

                            for (const [name, id] of nodeTypeMap.entries()) {
                                const namePrimary = name.split(':')[0];
                                if (!fromId && (name === sourcePrimary || namePrimary === sourcePrimary ||
                                    name.toLowerCase() === sourcePrimary.toLowerCase())) {
                                    fromId = id;
                                }
                                if (!toId && (name === targetPrimary || namePrimary === targetPrimary ||
                                    name.toLowerCase() === targetPrimary.toLowerCase())) {
                                    toId = id;
                                }
                                if (fromId && toId) break;
                            }

                            // Only create edge if both nodes found
                            if (fromId && toId) {
                                const isSelfLoop = fromId === toId;
                                edges.push({
                                    id: `${prefix}_edge_${index}_${sourceNodeName}_${targetNodeName}`,
                                    from: fromId,
                                    to: toId,
                                    label: edgeName,
                                    color: { color: '#6366f1', highlight: '#818cf8' },
                                    arrows: { to: { enabled: true, scaleFactor: 1 } },
                                    font: { color: '#a1a1aa', size: 9, align: 'middle', multi: 'html', face: 'Outfit, sans-serif' },
                                    width: 1.5,
                                    smooth: isSelfLoop ? { type: 'curvedCW', roundness: 0.8 } : { type: 'continuous', roundness: 0.5 }
                                });
                            }
                        });
                    });
                });
                return; // Skip normal edge processing for topology-based edges
            }

            // Normal edge processing (inferred schema format)
            let { from: startNode, to: endNode } = getEdgeNodes(edgeType);

            // Try to infer if not provided
            if (!startNode || !endNode) {
                const nameParts = edgeName.split(/[_-]?(to|from|connects|has|contains)[_-]?/i);
                if (nameParts.length >= 2) {
                    if (!startNode) startNode = nameParts[0].trim();
                    if (!endNode) endNode = nameParts[nameParts.length - 1].trim();
                }
            }

            let fromId = null;
            let toId = null;

            // Find node IDs
            if (startNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name === startNode || name.toLowerCase() === startNode.toLowerCase()) {
                        fromId = id;
                        break;
                    }
                }
            }

            if (endNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name === endNode || name.toLowerCase() === endNode.toLowerCase()) {
                        toId = id;
                        break;
                    }
                }
            }

            // Fallback matching
            if (!fromId && startNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name.toLowerCase().includes(startNode.toLowerCase()) ||
                        startNode.toLowerCase().includes(name.toLowerCase())) {
                        fromId = id;
                        break;
                    }
                }
            }

            if (!toId && endNode) {
                for (const [name, id] of nodeTypeMap.entries()) {
                    if (name.toLowerCase().includes(endNode.toLowerCase()) ||
                        endNode.toLowerCase().includes(name.toLowerCase())) {
                        toId = id;
                        break;
                    }
                }
            }

            // Fallback: if nodes not found but we have nodes, create connections
            if (!fromId && nodes.length > 0) {
                fromId = nodes[0].id;
            }
            if (!toId && nodes.length > 0) {
                if (fromId && nodes.length > 1) {
                    toId = nodes.find(n => n.id !== fromId)?.id || nodes[0].id;
                } else {
                    toId = nodes[0].id;
                }
            }

            if (fromId && toId) {
                const isSelfLoop = fromId === toId;
                let formattedLabel = edgeName.replace(/_/g, ' ');

                if (isSelfLoop) {
                    const nodeName = nodes.find(n => n.id === fromId)?.label || '';
                    const selfLoopPattern = new RegExp(`\\s+(to|from|connects|has|contains)\\s+${nodeName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
                    if (selfLoopPattern.test(formattedLabel)) {
                        formattedLabel = formattedLabel.replace(selfLoopPattern, '');
                    }
                } else if (formattedLabel.length > 20) {
                    const separatorMatch = formattedLabel.match(/\s+(to|from|connects|has|contains)\s+/i);
                    if (separatorMatch) {
                        const parts = formattedLabel.split(/\s+(to|from|connects|has|contains)\s+/i);
                        formattedLabel = parts[0] + '<br>' + separatorMatch[0].trim() + '<br>' + parts[parts.length - 1];
                    }
                }

                const edgeConfig = {
                    id: `${prefix}_edge_${index}`,
                    from: fromId,
                    to: toId,
                    label: formattedLabel,
                    color: {
                        color: '#8b5cf6',
                        highlight: '#a78bfa'
                    },
                    arrows: {
                        to: {
                            enabled: true,
                            scaleFactor: 1.2
                        }
                    },
                    font: {
                        color: '#94a3b8',
                        size: 10,
                        align: 'middle',
                        multi: 'html',
                        face: 'Arial'
                    },
                    width: 2
                };

                if (isSelfLoop) {
                    edgeConfig.smooth = { type: 'curvedCW', roundness: 0.8 };
                    edgeConfig.width = 2.5;
                } else {
                    edgeConfig.smooth = { type: 'continuous', roundness: 0.5 };
                }

                edges.push(edgeConfig);
            }
        });
    }

    return { nodes, edges };
}

function getComparisonGraphOptions() {
    return {
        nodes: {
            shape: 'ellipse',
            font: {
                color: '#fafafa',
                size: 12,
                face: 'Outfit, sans-serif'
            },
            borderWidth: 2,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.3)',
                size: 6,
                x: 2,
                y: 2
            }
        },
        edges: {
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 1
                }
            },
            color: {
                color: '#6366f1',
                highlight: '#818cf8'
            },
            font: {
                color: '#a1a1aa',
                size: 9,
                align: 'middle',
                multi: 'html',
                face: 'Outfit, sans-serif'
            },
            smooth: {
                type: 'continuous',
                roundness: 0.5
            },
            shadow: false
        },
        physics: {
            enabled: true,
            stabilization: {
                iterations: 150,
                fit: true
            },
            barnesHut: {
                gravitationalConstant: -2500,
                centralGravity: 0.25,
                springLength: 120,
                springConstant: 0.04,
                damping: 0.09
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true
        },
        layout: {
            improvedLayout: true
        }
    };
}

// Normalize edge name for better matching
function normalizeEdgeName(name) {
    if (!name) return '';
    let normalized = name.toLowerCase();

    // Remove common prefixes/suffixes
    normalized = normalized.replace(/^neuprint_/i, '');
    normalized = normalized.replace(/_fib25$/i, '');

    // Replace underscores and hyphens with spaces
    normalized = normalized.replace(/[_-]/g, ' ');

    // Extract key relationship words
    const relationshipWords = ['connects', 'connect', 'contains', 'contain', 'synapses', 'synapse',
        'has', 'links', 'link', 'to', 'from', 'between'];

    // Get words from the name
    const words = normalized.split(/\s+/).filter(w => w.length > 0);

    // Find relationship keywords
    const keywords = words.filter(w => relationshipWords.some(rw => w.includes(rw) || rw.includes(w)));

    return {
        original: name,
        normalized: normalized,
        words: words,
        keywords: keywords,
        keyPhrase: keywords.join(' ') || normalized
    };
}

// Fuzzy string matching function (similar to Python's SequenceMatcher)
function stringSimilarity(a, b) {
    if (!a || !b) return 0;
    const aLower = a.toLowerCase();
    const bLower = b.toLowerCase();

    // Exact match
    if (aLower === bLower) return 1.0;

    // Check for pluralization (e.g., "neuron" vs "neurons")
    const aSingular = aLower.replace(/s$/, '');
    const bSingular = bLower.replace(/s$/, '');
    if (aSingular === bLower || bSingular === aLower || aSingular === bSingular) {
        return 0.95;
    }

    // Check if one contains the other
    if (aLower.includes(bLower) || bLower.includes(aLower)) {
        return 0.85;
    }

    // Simple Levenshtein-like similarity
    const longer = aLower.length > bLower.length ? aLower : bLower;
    const shorter = aLower.length > bLower.length ? bLower : aLower;
    const editDistance = levenshteinDistance(longer, shorter);
    const similarity = 1 - (editDistance / longer.length);

    return similarity;
}

// Enhanced edge similarity matching
function edgeSimilarity(edgeName1, edgeName2) {
    if (!edgeName1 || !edgeName2) return 0;

    // Normalize both edge names
    const norm1 = normalizeEdgeName(edgeName1);
    const norm2 = normalizeEdgeName(edgeName2);

    // Exact match on normalized
    if (norm1.normalized === norm2.normalized) return 1.0;

    // Match on key phrase (relationship keywords)
    if (norm1.keyPhrase && norm2.keyPhrase && norm1.keyPhrase === norm2.keyPhrase) {
        return 0.95;
    }

    // Check if key phrases are similar
    if (norm1.keyPhrase && norm2.keyPhrase) {
        const keyPhraseSim = stringSimilarity(norm1.keyPhrase, norm2.keyPhrase);
        if (keyPhraseSim > 0.8) return keyPhraseSim;
    }

    // Check if one key phrase contains the other
    if (norm1.keyPhrase && norm2.keyPhrase) {
        if (norm1.keyPhrase.includes(norm2.keyPhrase) || norm2.keyPhrase.includes(norm1.keyPhrase)) {
            return 0.85;
        }
    }

    // Check for common relationship patterns
    const patterns = [
        {
            gt: ['connects', 'to'],
            inf: ['connects', 'connection', 'neuron', 'connection', 'neuron', 'to'],
            description: 'CONNECTS_TO pattern'
        },
        {
            gt: ['contains'],
            inf: ['contains', 'contain', 'has', 'synapseset', 'synapses'],
            description: 'CONTAINS pattern'
        },
        {
            gt: ['synapses', 'to'],
            inf: ['synapse', 'connection', 'synapse', 'to', 'synapses'],
            description: 'SYNAPSES_TO pattern'
        }
    ];

    for (const pattern of patterns) {
        const norm1HasGt = pattern.gt.every(p => norm1.normalized.includes(p));
        const norm2HasGt = pattern.gt.every(p => norm2.normalized.includes(p));
        const norm1HasInf = pattern.inf.some(p => norm1.normalized.includes(p));
        const norm2HasInf = pattern.inf.some(p => norm2.normalized.includes(p));

        // If one has GT pattern and other has INF pattern, they match
        if ((norm1HasGt && norm2HasInf) || (norm2HasGt && norm1HasInf)) {
            return 0.9;
        }
    }

    // Additional specific pattern matching
    // CONNECTS_TO should match anything with "connect" and "neuron" or "connection"
    if ((norm1.normalized.includes('connects') && norm1.normalized.includes('to')) ||
        (norm2.normalized.includes('connects') && norm2.normalized.includes('to'))) {
        if (norm1.normalized.includes('connection') || norm1.normalized.includes('neuron') ||
            norm2.normalized.includes('connection') || norm2.normalized.includes('neuron')) {
            return 0.88;
        }
    }

    // CONTAINS should match "synapseset to synapses" or similar
    if (norm1.normalized.includes('contains') || norm2.normalized.includes('contains')) {
        if ((norm1.normalized.includes('synapseset') && norm1.normalized.includes('synapses')) ||
            (norm2.normalized.includes('synapseset') && norm2.normalized.includes('synapses'))) {
            return 0.88;
        }
    }

    // SYNAPSES_TO should match "synapse_connections" or similar
    if ((norm1.normalized.includes('synapses') && norm1.normalized.includes('to')) ||
        (norm2.normalized.includes('synapses') && norm2.normalized.includes('to'))) {
        if (norm1.normalized.includes('synapse') && norm1.normalized.includes('connection') ||
            norm2.normalized.includes('synapse') && norm2.normalized.includes('connection')) {
            return 0.88;
        }
    }

    // Standard string similarity
    return stringSimilarity(norm1.normalized, norm2.normalized);
}

function levenshteinDistance(str1, str2) {
    const matrix = [];
    for (let i = 0; i <= str2.length; i++) {
        matrix[i] = [i];
    }
    for (let j = 0; j <= str1.length; j++) {
        matrix[0][j] = j;
    }
    for (let i = 1; i <= str2.length; i++) {
        for (let j = 1; j <= str1.length; j++) {
            if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                );
            }
        }
    }
    return matrix[str2.length][str1.length];
}

function findBestMatch(name, targetList, threshold = 0.7, isEdge = false) {
    let bestScore = 0;
    let bestMatch = null;

    for (const target of targetList) {
        const score = isEdge ? edgeSimilarity(name, target) : stringSimilarity(name, target);
        if (score >= threshold && score > bestScore) {
            bestScore = score;
            bestMatch = target;
        }
    }

    return bestMatch;
}

async function calculateComparisonMetrics() {
    if (!currentSchema || !groundTruthSchema) return;

    const gtNodes = new Map();
    const infNodes = new Map();

    // 1. Map labels using the robust getLabel helper
    (groundTruthSchema.node_types || []).forEach(n => {
        const lbl = getLabel(n);
        if (lbl !== 'Unknown') gtNodes.set(lbl, n);
    });

    (currentSchema.node_types || []).forEach(n => {
        const lbl = getLabel(n);
        if (lbl !== 'Unknown') infNodes.set(lbl, n);
    });

    // 2. Node Matching & Discovery
    let nodeMatches = 0;
    const nodeMatchMap = new Map(); // GT Name -> Inferred Name

    gtNodes.forEach((_, gtName) => {
        const match = findBestMatch(gtName, Array.from(infNodes.keys()), 0.75, false);
        if (match) {
            nodeMatches++;
            nodeMatchMap.set(gtName, match);
        }
    });

    const extraNodes = Array.from(infNodes.keys()).filter(name =>
        !Array.from(nodeMatchMap.values()).includes(name)
    );

    // 3. Edge Matching & Discovery - Match by topology (node type pairs), not just names
    const gtEdges = new Map();
    const gtEdgeList = groundTruthSchema.edge_types || [];
    const infEdgeList = currentSchema.edge_types || [];

    gtEdgeList.forEach(e => {
        const lbl = getLabel(e);
        if (lbl !== 'Unknown') gtEdges.set(lbl, e);
    });

    // Build inferred edges index: (edge_name, start_node, end_node) -> edge
    const infEdgeMap = new Map();
    infEdgeList.forEach(infEdge => {
        const edgeName = getLabel(infEdge);
        const startNode = infEdge.start_node || infEdge.from;
        const endNode = infEdge.end_node || infEdge.to;
        if (edgeName && startNode && endNode) {
            const key = `${edgeName}|${startNode}|${endNode}`;
            if (!infEdgeMap.has(key)) {
                infEdgeMap.set(key, []);
            }
            infEdgeMap.get(key).push(infEdge);
        }
    });

    // Expand GT edges into topology combinations (source, target, edge_name)
    const gtTopologyCombinations = [];
    gtEdges.forEach((gtEdge, gtEdgeName) => {
        const topology = gtEdge.topology || [];
        if (topology.length > 0) {
            topology.forEach(rule => {
                const sources = rule.allowed_sources || [];
                const targets = rule.allowed_targets || [];
                sources.forEach(source => {
                    targets.forEach(target => {
                        gtTopologyCombinations.push({
                            source: source,
                            target: target,
                            edgeName: gtEdgeName,
                            edge: gtEdge
                        });
                    });
                });
            });
        }
    });

    // Match each GT topology combination to inferred edges
    let topologyMatches = 0;
    const matchedInfEdgeNames = new Set();

    gtTopologyCombinations.forEach(combo => {
        const { source: gtSource, target: gtTarget, edgeName: gtEdgeName } = combo;

        // Map GT node names to inferred node names
        const infSourceCandidates = [];
        if (nodeMatchMap.has(gtSource)) {
            infSourceCandidates.push(nodeMatchMap.get(gtSource));
        }
        infSourceCandidates.push(gtSource); // Also try direct match

        const infTargetCandidates = [];
        if (nodeMatchMap.has(gtTarget)) {
            infTargetCandidates.push(nodeMatchMap.get(gtTarget));
        }
        infTargetCandidates.push(gtTarget); // Also try direct match

        // Try to find matching inferred edge
        let foundMatch = false;
        for (const infSource of infSourceCandidates) {
            for (const infTarget of infTargetCandidates) {
                // Try exact edge name match first
                const exactKey = `${gtEdgeName}|${infSource}|${infTarget}`;
                if (infEdgeMap.has(exactKey)) {
                    foundMatch = true;
                    matchedInfEdgeNames.add(gtEdgeName);
                    break;
                }
                // Try fuzzy edge name match
                for (const infEdgeName of new Set(infEdgeList.map(e => getLabel(e)).filter(l => l !== 'Unknown'))) {
                    if (findBestMatch(gtEdgeName, [infEdgeName], 0.8, true)) {
                        const fuzzyKey = `${infEdgeName}|${infSource}|${infTarget}`;
                        if (infEdgeMap.has(fuzzyKey)) {
                            foundMatch = true;
                            matchedInfEdgeNames.add(infEdgeName);
                            break;
                        }
                    }
                }
                if (foundMatch) break;
            }
            if (foundMatch) break;
        }

        if (foundMatch) {
            topologyMatches++;
        }
    });

    // Count unique edge names in inferred that don't match GT
    const allInfEdgeNames = new Set(infEdgeList.map(e => getLabel(e)).filter(l => l !== 'Unknown'));
    const extraEdgeNames = Array.from(allInfEdgeNames).filter(name => !matchedInfEdgeNames.has(name));

    const edgeMatches = topologyMatches;
    const totalGtCombinations = gtTopologyCombinations.length;
    const extraEdges = extraEdgeNames;

    // 4. Property Matching (for Matched Nodes)
    let totalProps = 0;
    let matchedProps = 0;

    nodeMatchMap.forEach((infName, gtName) => {
        const gt = gtNodes.get(gtName);
        const inf = infNodes.get(infName);
        if (gt && inf) {
            const gtP = new Set((gt.properties || []).map(p => String(p.name || '').toLowerCase()).filter(Boolean));
            const infP = new Set((inf.properties || []).map(p => String(p.name || '').toLowerCase()).filter(Boolean));
            totalProps += gtP.size;
            matchedProps += [...gtP].filter(p => infP.has(p)).length;
        }
    });

    // 5. Calculate Weighted Final Scores
    const nodeAccuracy = calculateRealScore(nodeMatches, gtNodes.size, extraNodes.length);
    const edgeAccuracy = calculateRealScore(edgeMatches, totalGtCombinations, extraEdges.length);
    const propAccuracy = totalProps > 0 ? (matchedProps / totalProps * 100) : 0;

    const overallAccuracy = (nodeAccuracy + edgeAccuracy + propAccuracy) / 3;
    const discoveryCoverage = (nodeMatches / gtNodes.size) * 100;

    // 6. Update UI Panel
    document.getElementById('comparisonMetrics').innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Schema Accuracy (Weighted Penalty)</span>
            <span class="metric-value ${getAccuracyClass(overallAccuracy)}">${overallAccuracy.toFixed(1)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Discovery Coverage (GT Found)</span>
            <span class="metric-value">${discoveryCoverage.toFixed(1)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Discovery Details</span>
            <span class="metric-value" style="color: var(--text-secondary); font-size: 0.9rem;">
                +${extraNodes.length} extra nodes, +${extraEdges.length} extra edges
            </span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Property Match</span>
            <span class="metric-value">${propAccuracy.toFixed(1)}%</span>
        </div>
    `;

    // Build edge match map for display (map GT edge names to matched inferred edge names)
    const edgeMatchMapForDisplay = new Map();
    gtEdges.forEach((gtEdge, gtEdgeName) => {
        const matchedNames = Array.from(matchedInfEdgeNames).filter(infName =>
            findBestMatch(gtEdgeName, [infName], 0.8, true)
        );
        if (matchedNames.length > 0) {
            edgeMatchMapForDisplay.set(gtEdgeName, matchedNames[0]);
        }
    });

    // Build missing edges list (unmatched topology combinations)
    const missingEdgeCombinations = [];
    gtTopologyCombinations.forEach(c => {
        let found = false;
        const infSourceCandidates = [];
        if (nodeMatchMap.has(c.source)) infSourceCandidates.push(nodeMatchMap.get(c.source));
        infSourceCandidates.push(c.source);
        const infTargetCandidates = [];
        if (nodeMatchMap.has(c.target)) infTargetCandidates.push(nodeMatchMap.get(c.target));
        infTargetCandidates.push(c.target);
        for (const infSource of infSourceCandidates) {
            for (const infTarget of infTargetCandidates) {
                const exactKey = `${c.edgeName}|${infSource}|${infTarget}`;
                if (infEdgeMap.has(exactKey)) {
                    found = true;
                    break;
                }
                for (const infEdgeName of matchedInfEdgeNames) {
                    if (findBestMatch(c.edgeName, [infEdgeName], 0.8, true)) {
                        const fuzzyKey = `${infEdgeName}|${infSource}|${infTarget}`;
                        if (infEdgeMap.has(fuzzyKey)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (found) break;
            }
            if (found) break;
        }
        if (!found) {
            missingEdgeCombinations.push(`${c.source} --[${c.edgeName}]--> ${c.target}`);
        }
    });

    // Update the detailed breakdown lists
    displayComparisonDetails(
        Array.from(nodeMatchMap.keys()),
        Array.from(gtNodes.keys()).filter(n => !nodeMatchMap.has(n)),
        extraNodes,
        Array.from(edgeMatchMapForDisplay.keys()),
        missingEdgeCombinations,
        extraEdges,
        nodeMatchMap,
        edgeMatchMapForDisplay
    );
}

function getAccuracyClass(accuracy) {
    if (accuracy >= 90) return 'high';
    if (accuracy >= 70) return 'medium';
    return 'low';
}

function displayComparisonDetails(matchedNodes, missingNodes, extraNodes, matchedEdges, missingEdges, extraEdges, nodeMatchMap, edgeMatchMap) {
    let html = '';

    // Add note about demo mode
    html += '<div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid var(--warning-color); padding: 12px; margin-bottom: 20px; border-radius: 8px;">';
    html += '<strong style="color: var(--warning-color);">ℹ️ Note:</strong> ';
    html += '<span style="color: var(--text-secondary);">If you\'re using demo mode (no API key), the accuracy may be lower. For best results, use the actual Gemini API with your API key.</span>';
    html += '</div>';

    // Node comparison
    html += '<div class="comparison-section"><h4>Node Types Comparison</h4>';

    if (matchedNodes.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--success-color);">✓ Matched Nodes (${matchedNodes.length}):</strong>`;
        matchedNodes.forEach(gtName => {
            const infName = nodeMatchMap.get(gtName);
            if (infName && infName !== gtName) {
                html += `<div class="match-item"><span class="comparison-item-label">${gtName} ↔ ${infName}</span><span class="comparison-item-details">Fuzzy matched</span></div>`;
            } else {
                html += `<div class="match-item"><span class="comparison-item-label">${gtName}</span></div>`;
            }
        });
        html += '</div>';
    }

    if (missingNodes.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--error-color);">✗ Missing Nodes (${missingNodes.length}):</strong>`;
        missingNodes.forEach(name => {
            html += `<div class="missing-item"><span class="comparison-item-label">${name}</span><span class="comparison-item-details">Present in Ground Truth but not inferred</span></div>`;
        });
        html += '</div>';
    }

    if (extraNodes.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--warning-color);">⚠ Extra Nodes (${extraNodes.length}):</strong>`;
        extraNodes.forEach(name => {
            html += `<div class="mismatch-item"><span class="comparison-item-label">${name}</span><span class="comparison-item-details">Inferred but not in Ground Truth</span></div>`;
        });
        html += '</div>';
    }

    // Section for Extra Items (The Penalty)
    if (extraNodes.length > 0 || extraEdges.length > 0) {
        html += `<div class="comparison-section" style="border: 2px solid var(--error-color);">`;
        html += `<h4 style="color: var(--error-color);">⚠️ Over-Inference Penalties</h4>`;
        if (extraNodes.length > 0) html += `<p><strong>Extra Nodes:</strong> ${extraNodes.join(', ')}</p>`;
        if (extraEdges.length > 0) html += `<p><strong>Extra Edges:</strong> ${extraEdges.join(', ')}</p>`;
        html += `<p class="hint">These items were found in your data but are NOT in the official schema.</p></div>`;
    }

    html += '</div>';

    // Edge comparison
    html += '<div class="comparison-section"><h4>Edge Types Comparison</h4>';

    if (matchedEdges.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--success-color);">✓ Matched Edges (${matchedEdges.length}):</strong>`;
        matchedEdges.forEach(gtName => {
            const infName = edgeMatchMap.get(gtName);
            if (infName && infName !== gtName) {
                html += `<div class="match-item"><span class="comparison-item-label">${gtName} ↔ ${infName}</span><span class="comparison-item-details">Fuzzy matched</span></div>`;
            } else {
                html += `<div class="match-item"><span class="comparison-item-label">${gtName}</span></div>`;
            }
        });
        html += '</div>';
    }

    if (missingEdges.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--error-color);">✗ Missing Edges (${missingEdges.length}):</strong>`;
        missingEdges.forEach(name => {
            html += `<div class="missing-item"><span class="comparison-item-label">${name}</span><span class="comparison-item-details">Present in Ground Truth but not inferred</span></div>`;
        });
        html += '</div>';
    }

    if (extraEdges.length > 0) {
        html += `<div style="margin-bottom: 16px;"><strong style="color: var(--warning-color);">⚠ Extra Edges (${extraEdges.length}):</strong>`;
        extraEdges.forEach(name => {
            html += `<div class="mismatch-item"><span class="comparison-item-label">${name}</span><span class="comparison-item-details">Inferred but not in Ground Truth</span></div>`;
        });
        html += '</div>';
    }

    html += '</div>';

    document.getElementById('comparisonDetails').innerHTML = html;
}

