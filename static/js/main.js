let currentJobId = null;
let statusCheckInterval = null;

// File input handling
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const fileList = document.getElementById('fileList');
const uploadForm = document.getElementById('uploadForm');

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    fileInput.files = files;
    updateFileList();
});

fileInput.addEventListener('change', updateFileList);

function updateFileList() {
    fileList.innerHTML = '';
    const files = Array.from(fileInput.files);
    
    if (files.length === 0) return;
    
    // Show count with emphasis
    const countHeader = document.createElement('div');
    countHeader.style.cssText = 'margin-bottom: 12px; font-weight: 700; color: var(--primary-color); font-size: 1.1rem; padding: 8px; background: rgba(99, 102, 241, 0.1); border-radius: 8px; border-left: 3px solid var(--primary-color);';
    countHeader.textContent = `✓ Selected ${files.length} file${files.length > 1 ? 's' : ''} - Ready to process together!`;
    fileList.appendChild(countHeader);
    
    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: var(--bg-color); border-radius: 8px; margin-bottom: 8px; border: 1px solid var(--border-color);';
        fileItem.innerHTML = `
            <span style="color: var(--text-primary);">${index + 1}. ${file.name}</span>
            <span class="file-size" style="color: var(--text-secondary); font-size: 0.9rem;">${formatFileSize(file.size)}</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const files = fileInput.files;
    if (files.length === 0) {
        alert('Please select at least one CSV file');
        return;
    }
    
    // Log multiple files
    if (files.length > 1) {
        console.log(`✓ Uploading ${files.length} files together:`, Array.from(files).map(f => f.name));
    }
    
    const formData = new FormData();
    Array.from(files).forEach(file => {
        formData.append('files[]', file);
    });
    
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Uploading...</span>';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            if (response.status === 413) {
                throw new Error('File size too large! Total upload must be less than 500MB. Try uploading fewer files at once.');
            }
            const text = await response.text();
            throw new Error(`Server error (${response.status}): ${text.substring(0, 100)}`);
        }
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        currentJobId = data.job_id;
        showProgress();
        startStatusCheck();
        
    } catch (error) {
        showError(error.message);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Start Schema Discovery</span>';
    }
});

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
            
            updateProgress(data);
            
            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(statusCheckInterval);
                
                if (data.status === 'completed') {
                    showResults(data.result);
                } else {
                    showError(data.message || 'An error occurred');
                }
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }, 2000); // Check every 2 seconds
}

function updateProgress(data) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusMessage = document.getElementById('statusMessage');
    
    progressFill.style.width = `${data.progress}%`;
    progressText.textContent = `${data.progress}%`;
    statusMessage.textContent = data.message || '';
}

// UI State Management
function showProgress() {
    document.getElementById('uploadCard').classList.add('hidden');
    document.getElementById('progressCard').classList.remove('hidden');
    document.getElementById('resultsCard').classList.add('hidden');
    document.getElementById('errorCard').classList.add('hidden');
}

function showResults(schema) {
    document.getElementById('progressCard').classList.add('hidden');
    document.getElementById('resultsCard').classList.remove('hidden');
    
    renderSchema(schema);
    
    // Setup download button
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.onclick = () => {
        window.location.href = `/download/${currentJobId}`;
    };
}

function showError(message) {
    document.getElementById('progressCard').classList.add('hidden');
    document.getElementById('errorCard').classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

function resetForm() {
    currentJobId = null;
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    document.getElementById('uploadCard').classList.remove('hidden');
    document.getElementById('errorCard').classList.add('hidden');
    document.getElementById('resultsCard').classList.add('hidden');
    fileInput.value = '';
    fileList.innerHTML = '';
    
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span>Start Schema Discovery</span>';
}

// Schema rendering
function renderSchema(schema) {
    const nodeTypesDiv = document.getElementById('nodeTypes');
    const edgeTypesDiv = document.getElementById('edgeTypes');
    
    nodeTypesDiv.innerHTML = '';
    edgeTypesDiv.innerHTML = '';
    
    // Render Node Types
    if (schema.node_types && schema.node_types.length > 0) {
        schema.node_types.forEach(nodeType => {
            const nodeItem = createSchemaItem(
                nodeType.name || nodeType.labels?.[0] || 'Unknown',
                nodeType.properties || [],
                'node'
            );
            nodeTypesDiv.appendChild(nodeItem);
        });
    } else {
        nodeTypesDiv.innerHTML = '<p style="color: var(--text-secondary);">No node types found</p>';
    }
    
    // Render Edge Types
    if (schema.edge_types && schema.edge_types.length > 0) {
        schema.edge_types.forEach(edgeType => {
            const edgeItem = createSchemaItem(
                edgeType.type || edgeType.name || 'Unknown',
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

// New Analysis button
document.getElementById('newAnalysisBtn').addEventListener('click', resetForm);

