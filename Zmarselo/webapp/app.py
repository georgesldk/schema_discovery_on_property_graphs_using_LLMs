import os
import json
import random
import threading
import time
import shutil
import subprocess
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
import sys
from pathlib import Path

# Add src to path so we can import pg_schema_llm
webapp_dir = Path(__file__).parent
src_dir = webapp_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import from the new pg_schema_llm system
from pg_schema_llm.io import build_graph
from pg_schema_llm.profiling import (
    profile_node_type,
    profile_edge_type,
)

# generate_logical_relationship_summary doesn't exist for graph-based, 
# so we create a simple wrapper that returns empty string
def generate_logical_relationship_summary(G):
    """Simple wrapper - graph-based logical summary not available in new system"""
    # For now, return empty string. Can be enhanced later if needed.
    return ""

# Import comparison functions
from difflib import SequenceMatcher

# Helper function to extract JSON from text (simple version)
def extract_json(text):
    """Extract JSON from text, handling markdown code blocks"""
    if not text:
        return None
    try:
        cleaned = text.strip()
        # Remove markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception:
        return None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size (increased for large datasets)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global state for job tracking
jobs = {}

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_uploads(upload_folder, schema_folder, keep_count=5):
    """
    Keep only the most recent N upload directories and schema_found directories.
    Deletes older directories to save disk space.
    """
    try:
        # Clean up upload directories
        if os.path.exists(upload_folder):
            upload_dirs = [d for d in os.listdir(upload_folder) if d.startswith('job_') and os.path.isdir(os.path.join(upload_folder, d))]
            if len(upload_dirs) > keep_count:
                # Sort by modification time (newest first)
                upload_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(upload_folder, d)), reverse=True)
                # Keep only the most recent ones
                to_delete = upload_dirs[keep_count:]
                for dir_name in to_delete:
                    dir_path = os.path.join(upload_folder, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted old upload directory: {dir_path}")
                    except Exception as e:
                        print(f"Error deleting {dir_path}: {e}")
        
        # Clean up schema_found directories
        if os.path.exists(schema_folder):
            schema_dirs = [d for d in os.listdir(schema_folder) if d.startswith('job_') and os.path.isdir(os.path.join(schema_folder, d))]
            if len(schema_dirs) > keep_count:
                # Sort by modification time (newest first)
                schema_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(schema_folder, d)), reverse=True)
                # Keep only the most recent ones
                to_delete = schema_dirs[keep_count:]
                for dir_name in to_delete:
                    dir_path = os.path.join(schema_folder, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted old schema directory: {dir_path}")
                    except Exception as e:
                        print(f"Error deleting {dir_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('schema_found', exist_ok=True)
os.makedirs('gt_schema', exist_ok=True)

# Clean up old uploads on startup (keep only last 5)
cleanup_old_uploads(app.config['UPLOAD_FOLDER'], 'schema_found', keep_count=5)

# profile_node_type and profile_edge_type are now imported from main.py

def generate_mock_schema(G, job_id):
    """Generate a mock schema from the graph structure for demo purposes"""
    jobs[job_id]['status'] = 'calling_api'
    jobs[job_id]['message'] = 'Generating mock schema (DEMO MODE - No API key required)...'
    time.sleep(2)  # Simulate API call delay
    
    node_types_list = []
    edge_types_list = []
    
    # Extract node types
    node_types = set(nx.get_node_attributes(G, 'node_type').values())
    for nt in node_types:
        nodes_of_type = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == nt]
        if not nodes_of_type:
            continue
        
        # Get properties from sample nodes
        sample_node = nodes_of_type[0]
        node_attrs = G.nodes[sample_node]
        properties = []
        
        for key, value in node_attrs.items():
            if key == 'node_type':
                continue
            prop_type = 'String'
            if isinstance(value, (int, float)):
                prop_type = 'Long' if isinstance(value, int) else 'Double'
            elif isinstance(value, bool):
                prop_type = 'Boolean'
            
            # Check if mandatory (>90% have this property)
            has_prop_count = sum(1 for n in nodes_of_type if key in G.nodes[n])
            mandatory = (has_prop_count / len(nodes_of_type)) > 0.9
            
            properties.append({
                'name': key,
                'type': prop_type,
                'mandatory': mandatory
            })
        
        node_types_list.append({
            'name': nt,
            'labels': [nt],
            'properties': properties
        })
    
    # Extract edge types
    edge_types = set(nx.get_edge_attributes(G, 'type').values())
    for et in edge_types:
        edges_of_type = [(u, v, attr) for u, v, attr in G.edges(data=True) if attr.get('type') == et]
        if not edges_of_type:
            continue
        
        # Get properties from sample edge
        sample_edge = edges_of_type[0]
        edge_attrs = sample_edge[2]
        properties = []
        
        for key, value in edge_attrs.items():
            if key == 'type':
                continue
            prop_type = 'String'
            if isinstance(value, (int, float)):
                prop_type = 'Long' if isinstance(value, int) else 'Double'
            elif isinstance(value, bool):
                prop_type = 'Boolean'
            
            # Check if mandatory
            has_prop_count = sum(1 for _, _, attr in edges_of_type if key in attr)
            mandatory = (has_prop_count / len(edges_of_type)) > 0.9
            
            properties.append({
                'name': key,
                'type': prop_type,
                'mandatory': mandatory
            })
        
        edge_types_list.append({
            'type': et,
            'name': et,
            'properties': properties
        })
    
    return json.dumps({
        'node_types': node_types_list,
        'edge_types': edge_types_list
    }, indent=2)

def call_gemini_api(prompt, job_id, G=None, use_mock=False):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Use mock mode if no API key or explicitly requested
    if not api_key or use_mock:
        if G is not None:
            return generate_mock_schema(G, job_id)
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = 'Mock mode requires graph data. Please upload CSV files.'
            return None
    
    # Real API call
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        jobs[job_id]['status'] = 'calling_api'
        jobs[job_id]['message'] = 'Calling Gemini API...'
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        return response.text
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = f'API Error: {str(e)}'
        return None

# extract_json is now imported from main.py

def process_schema_discovery(data_dir, output_dir, job_id):
    """Process schema discovery in a background thread"""
    try:
        jobs[job_id]['status'] = 'building_graph'
        jobs[job_id]['message'] = 'Building graph from CSV files...'
        
        # Build Graph
        G = build_graph(data_dir)
        
        if G.number_of_nodes() == 0:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = 'Graph is empty. No data found.'
            return
        
        jobs[job_id]['status'] = 'profiling'
        jobs[job_id]['message'] = 'Profiling nodes and edges...'
        jobs[job_id]['progress'] = 30
        
        # Profile
        node_types = set(nx.get_node_attributes(G, 'node_type').values())
        edge_types = set(nx.get_edge_attributes(G, 'type').values())
        
        context_report = f"Total Nodes: {G.number_of_nodes()}\nTotal Edges: {G.number_of_edges()}\n"
        for nt in node_types: context_report += profile_node_type(G, nt)
        for et in edge_types: context_report += profile_edge_type(G, et)
        
        # Add logical relationship analysis
        logical_summary = generate_logical_relationship_summary(G)
        if logical_summary:
            context_report += logical_summary
        
        jobs[job_id]['progress'] = 60
        jobs[job_id]['status'] = 'generating_prompt'
        jobs[job_id]['message'] = 'Generating schema inference prompt...'
        
        # Enhanced prompt - same as main.py (dataset-agnostic Property Graph standards)
        prompt = f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a logical Property Graph schema from raw physical data structures, following industry-standard Property Graph principles.
    
    DATA PROFILE:
    {context_report}
    
    STRICT DATASET-AGNOSTIC PROPERTY GRAPH HEURISTICS:
    
    1. LOGICAL BYPASS RULE (Collapse Technical Intermediaries) - HIGHEST PRIORITY:
       - Technical Container nodes are grouping/collection mechanisms with minimal properties (typically < 3 meaningful properties beyond IDs).
       - Patterns: names containing "Set", "Collection", "Group", "Container", "Link", "Join", "Mapping", "Association".
       - MANDATORY ACTION: If the profile shows "Entity A -> TechnicalContainer -> Entity C" paths, you MUST:
         * Create a direct edge type: Entity A -> Entity C
         * Use appropriate edge label (CONNECTS_TO, CONTAINS, or SYNAPSES_TO based on semantic meaning)
         * DO NOT create edges that go through the technical container in your schema
       - If the profile includes "[LOGICAL RELATIONSHIP ANALYSIS]", those direct relationships are REQUIRED in your output.
       - The logical schema represents functional relationships, not physical storage artifacts.
       - Property Graphs prioritize direct semantic connections over intermediate technical structures.
    
    2. ENTITY CONSOLIDATION (Deduplicate Semantic Equivalents):
       - If multiple node types represent the same logical entity with different attribute sets, merge them into a single node type.
       - Properties from all variants should be merged, with "mandatory" set based on > 98% fill density.
       - Only keep truly distinct entity types that represent different concepts.
    
    3. STANDARDIZED EDGE LABELS (Property Graph Convention):
       - MANDATORY: Use ONLY these three edge labels - no exceptions:
         * CONNECTS_TO: For high-level structural/functional connections between major entities
         * CONTAINS: For parent-child or containment relationships (hierarchical)
         * SYNAPSES_TO: For fine-grained functional/operational links (use sparingly, only when CONNECTS_TO is too coarse)
       - DO NOT use: technical names (HAS_SET, LINKS_TO), action verbs (CREATES, DELETES), file-based names (FROM_CSV, TO_TABLE), or generic verbs (ASSOCIATED_WITH, DEPENDS_ON).
       - Consolidate all edges between the same two node types into ONE edge type using the appropriate standard label.
    
    4. SCHEMA NORMALIZATION:
       - NODE NAMING: Singular PascalCase (e.g., "Entity", "Item", "Category" - NOT "Entities", "EntityType").
       - PROPERTY TYPES: Use "String", "Long", "Double", "Boolean", "StringArray", "Point" (for spatial data).
       - MANDATORY FLAGS: Set "mandatory: true" ONLY for properties with > 98% fill density across all instances.
       - EDGE PROPERTIES: Include edge properties (e.g., weight, confidence, count) when they carry semantic meaning.
    
    5. TOPOLOGY REQUIREMENTS:
       - Each edge_type MUST specify "start_node" and "end_node" fields with the exact node type names.
       - Self-loops (same node type as source and target) are allowed and valid.
       - The schema should represent a logical graph, not a physical data model.
    
    CRITICAL: This is a Property Graph schema, not a relational model. Focus on:
    - Direct entity-to-entity relationships
    - Logical, not physical, structure  
    - Semantic clarity over technical accuracy
    - Standard naming conventions
    
    OUTPUT JSON FORMAT:
    {{
      "node_types": [
        {{
          "name": "NodeLabel",
          "properties": [
            {{"name": "propertyName", "type": "String|Long|Double|Boolean|StringArray|Point", "mandatory": true|false}}
          ]
        }}
      ],
      "edge_types": [
        {{
          "name": "CONNECTS_TO|CONTAINS|SYNAPSES_TO",
          "start_node": "SourceNodeLabel",
          "end_node": "TargetNodeLabel",
          "properties": [
            {{"name": "propertyName", "type": "String|Long|Double|Boolean", "mandatory": true|false}}
          ]
        }}
      ]
    }}
    """
        
        jobs[job_id]['progress'] = 70
        # Check if we should use mock mode
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        use_mock = not api_key
        
        response = call_gemini_api(prompt, job_id, G=G, use_mock=use_mock)
        
        if response:
            jobs[job_id]['progress'] = 90
            jobs[job_id]['status'] = 'saving'
            jobs[job_id]['message'] = 'Saving inferred schema...'
            
            schema = extract_json(response)
            if schema:
                output_file = os.path.join(output_dir, "inferred_schema.json")
                with open(output_file, "w") as f:
                    json.dump(schema, f, indent=4)
                
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['message'] = 'Schema discovery completed successfully!'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['result'] = schema
                jobs[job_id]['output_file'] = output_file
            else:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['message'] = 'Failed to parse JSON response from API'
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = 'API call failed'
            
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = f'Error: {str(e)}'

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File size too large. Total upload size must be less than 500MB. Please upload fewer files or compress them.'}), 413

@app.route('/process-dataset/<dataset_id>', methods=['POST'])
def process_dataset(dataset_id):
    """Process a proof-of-concept dataset using infer.py script"""
    # Available datasets
    available_datasets = ['starwars', 'pole', 'mb6', 'fib25', 'ldbc']
    
    if dataset_id not in available_datasets:
        return jsonify({'error': f'Dataset {dataset_id} not available. Available: {", ".join(available_datasets)}'}), 404
    
    # Create a unique job ID
    job_id = f"poc_{dataset_id}_{int(time.time() * 1000)}"
    
    # Initialize job with console output tracking
    jobs[job_id] = {
        'status': 'queued',
        'message': 'Job queued',
        'progress': 0,
        'dataset_id': dataset_id,
        'mode': 'proof_of_concept',
        'console_output': []  # Store console output lines
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=run_infer_script, args=(dataset_id, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': f'Processing dataset {dataset_id}...',
        'dataset_id': dataset_id
    })

def run_infer_script(dataset_id, job_id):
    """Run infer.py script and capture output"""
    try:
        # Get the script path
        webapp_dir = Path(__file__).parent
        scripts_dir = webapp_dir.parent / "scripts"
        infer_script = scripts_dir / "infer.py"
        
        if not infer_script.exists():
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = f'Script not found: {infer_script}'
            jobs[job_id]['console_output'].append(f'ERROR: Script not found: {infer_script}')
            return
        
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = f'Running infer.py for {dataset_id}...'
        jobs[job_id]['console_output'].append(f'>>> Starting schema inference for dataset: {dataset_id}')
        jobs[job_id]['console_output'].append(f'>>> Command: python scripts/infer.py {dataset_id}')
        jobs[job_id]['progress'] = 10
        
        # Change to Zmarselo directory
        zmarselo_dir = webapp_dir.parent
        os.chdir(str(zmarselo_dir))
        
        # Run the script and capture output in real-time
        process = subprocess.Popen(
            ['python', 'scripts/infer.py', dataset_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line:
                    jobs[job_id]['console_output'].append(line)
                    # Update progress based on keywords
                    if 'Building TypeStats' in line:
                        jobs[job_id]['progress'] = 20
                        jobs[job_id]['message'] = 'Building statistics...'
                    elif 'Asking Gemini' in line:
                        jobs[job_id]['progress'] = 60
                        jobs[job_id]['message'] = 'Calling Gemini API...'
                    elif 'Node types:' in line or 'Edge types:' in line:
                        jobs[job_id]['progress'] = 40
                    elif 'ERROR' in line.upper() or 'Error' in line:
                        jobs[job_id]['status'] = 'error'
                        jobs[job_id]['message'] = line
        
        process.wait()
        
        if process.returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['message'] = f'Schema inference completed for {dataset_id}'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['console_output'].append('>>> Schema inference completed successfully!')
            
            # Check if output file exists
            output_file = zmarselo_dir / f"03_outputs/schemas/inferred/{dataset_id}/inf_{dataset_id}.json"
            if output_file.exists():
                jobs[job_id]['output_file'] = str(output_file)
                jobs[job_id]['result'] = 'success'
            else:
                jobs[job_id]['console_output'].append(f'WARNING: Output file not found: {output_file}')
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = f'Script failed with return code {process.returncode}'
            jobs[job_id]['console_output'].append(f'>>> ERROR: Script failed with return code {process.returncode}')
            
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = f'Error running script: {str(e)}'
        jobs[job_id]['console_output'].append(f'>>> EXCEPTION: {str(e)}')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload and process a new dataset (New Dataset mode)"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create a unique job ID
    job_id = f"job_{int(time.time() * 1000)}"
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            saved_files.append(filename)
    
    if not saved_files:
        return jsonify({'error': 'No valid CSV files uploaded'}), 400
    
    # Initialize job
    jobs[job_id] = {
        'status': 'queued',
        'message': 'Job queued',
        'progress': 0,
        'upload_dir': upload_dir,
        'mode': 'new_dataset'
    }
    
    # Start processing in background thread
    output_dir = os.path.join('schema_found', job_id)
    os.makedirs(output_dir, exist_ok=True)
    thread = threading.Thread(target=process_schema_discovery, args=(upload_dir, output_dir, job_id))
    thread.daemon = True
    thread.start()
    
    # Clean up old uploads (keep only last 5)
    cleanup_old_uploads(app.config['UPLOAD_FOLDER'], 'schema_found', keep_count=5)
    
    return jsonify({
        'job_id': job_id,
        'message': f'Uploaded {len(saved_files)} file(s). Processing started.',
        'files': saved_files
    })

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'message': job.get('message', ''),
        'progress': job.get('progress', 0),
        'mode': job.get('mode', 'proof_of_concept'),  # Include mode information
        'dataset_id': job.get('dataset_id'),  # Include dataset_id for POC mode
        'console_output': job.get('console_output', [])  # Include console output
    }
    
    if job['status'] == 'completed' and 'result' in job:
        response['result'] = job['result']
        response['output_file'] = job.get('output_file', '')
    
    return jsonify(response)

@app.route('/download/<job_id>')
def download_result(job_id):
    if job_id not in jobs or jobs[job_id]['status'] != 'completed':
        return jsonify({'error': 'Result not available'}), 404
    
    output_file = jobs[job_id].get('output_file')
    if not output_file or not os.path.exists(output_file):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(output_file, as_attachment=True, download_name='inferred_schema.json')

@app.route('/api/load-schema')
def load_schema():
    """Load schema from output file"""
    file_path = request.args.get('file')
    if not file_path:
        return jsonify({'error': 'File path not provided'}), 400
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return jsonify(schema)
    except Exception as e:
        return jsonify({'error': f'Failed to load schema: {str(e)}'}), 500

@app.route('/compare-dataset/<dataset_id>', methods=['POST'])
def compare_dataset(dataset_id):
    """Run compare.py script for a dataset"""
    available_datasets = ['starwars', 'pole', 'mb6', 'fib25', 'ldbc']
    
    if dataset_id not in available_datasets:
        return jsonify({'error': f'Dataset {dataset_id} not available'}), 404
    
    # Create a unique job ID for compare
    job_id = f"compare_{dataset_id}_{int(time.time() * 1000)}"
    
    # Initialize job with console output tracking
    jobs[job_id] = {
        'status': 'queued',
        'message': 'Job queued',
        'progress': 0,
        'dataset_id': dataset_id,
        'mode': 'compare',
        'console_output': []
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=run_compare_script, args=(dataset_id, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': f'Comparing dataset {dataset_id}...',
        'dataset_id': dataset_id
    })

def run_compare_script(dataset_id, job_id):
    """Run compare.py script and capture output"""
    try:
        webapp_dir = Path(__file__).parent
        zmarselo_dir = webapp_dir.parent
        
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['message'] = f'Running compare.py for {dataset_id}...'
        jobs[job_id]['console_output'].append(f'>>> Starting comparison for dataset: {dataset_id}')
        jobs[job_id]['console_output'].append(f'>>> Command: python scripts/compare.py {dataset_id}')
        jobs[job_id]['progress'] = 10
        
        # Change to Zmarselo directory
        os.chdir(str(zmarselo_dir))
        
        # Run the script and capture output in real-time
        process = subprocess.Popen(
            ['python', 'scripts/compare.py', dataset_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line:
                    jobs[job_id]['console_output'].append(line)
                    # Update progress based on keywords
                    if 'NODE MATCHING' in line:
                        jobs[job_id]['progress'] = 30
                    elif 'EDGE LABEL MAPPING' in line:
                        jobs[job_id]['progress'] = 50
                    elif 'TOPOLOGY' in line:
                        jobs[job_id]['progress'] = 70
                    elif 'FINAL SCORES' in line:
                        jobs[job_id]['progress'] = 90
                    elif 'ERROR' in line.upper() or 'Error' in line:
                        jobs[job_id]['status'] = 'error'
                        jobs[job_id]['message'] = line
        
        process.wait()
        
        if process.returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['message'] = f'Comparison completed for {dataset_id}'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['console_output'].append('>>> Comparison completed successfully!')
            
            # Parse results from console output
            parse_compare_results(job_id)
        else:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['message'] = f'Script failed with return code {process.returncode}'
            jobs[job_id]['console_output'].append(f'>>> ERROR: Script failed with return code {process.returncode}')
            
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = f'Error running script: {str(e)}'
        jobs[job_id]['console_output'].append(f'>>> EXCEPTION: {str(e)}')

def parse_compare_results(job_id):
    """Parse compare.py output to extract structured results"""
    if job_id not in jobs:
        return
    
    output = jobs[job_id].get('console_output', [])
    results = {
        'nodes': {'gt': [], 'inferred': [], 'matches': []},
        'edges': {'gt': [], 'inferred': [], 'matches': []},
        'scores': {}
    }
    
    # Parse node matching
    in_gt_nodes = False
    in_inferred_nodes = False
    in_gt_edges = False
    in_inferred_edges = False
    in_scores_section = False
    
    for line in output:
        if '[ RAW GT NODES ]' in line:
            in_gt_nodes = True
            in_inferred_nodes = False
            continue
        elif '[ RAW INFERRED NODES ]' in line:
            in_gt_nodes = False
            in_inferred_nodes = True
            continue
        elif '[ NODE MATCHES ]' in line:
            in_gt_nodes = False
            in_inferred_nodes = False
            continue
        elif '[ RAW GT EDGE TYPES ]' in line:
            in_gt_nodes = False
            in_inferred_nodes = False
            in_gt_edges = True
            in_inferred_edges = False
            continue
        elif '[ RAW INFERRED EDGE TYPES ]' in line:
            in_gt_edges = False
            in_inferred_edges = True
            continue
        elif '[ EDGE LABEL MAP ]' in line:
            in_gt_edges = False
            in_inferred_edges = False
            continue
        elif '[ FINAL SCORES ]' in line:
            in_gt_edges = False
            in_inferred_edges = False
            in_scores_section = True
            continue
        
        if in_gt_nodes and '[ NODE ]' in line:
            node_name = line.split('[ NODE ]')[1].strip()
            if node_name:
                results['nodes']['gt'].append(node_name)
        elif in_inferred_nodes and '[ NODE ]' in line:
            node_name = line.split('[ NODE ]')[1].strip()
            if node_name:
                results['nodes']['inferred'].append(node_name)
        elif in_gt_edges and '[ EDGE ]' in line:
            edge_name = line.split('[ EDGE ]')[1].strip()
            if edge_name:
                results['edges']['gt'].append(edge_name)
        elif in_inferred_edges and '[ EDGE ]' in line:
            edge_name = line.split('[ EDGE ]')[1].strip()
            if edge_name:
                results['edges']['inferred'].append(edge_name)
        elif '[ MAP ]' in line and '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                gt = parts[0].split('[ MAP ]')[1].strip()
                inferred = parts[1].strip()
                if ':' in gt:  # Edge match (format: "EDGE_NAME: source -> target")
                    results['edges']['matches'].append({'gt': gt, 'inferred': inferred})
                else:  # Node match
                    results['nodes']['matches'].append({'gt': gt, 'inferred': inferred})
        elif in_scores_section and '[' in line and ']' in line and '%' in line:
            # Parse scores like "[ NODE ACCURACY ] 100.00%"
            parts = line.split(']')
            if len(parts) == 2:
                metric = parts[0].replace('[', '').strip()
                value = parts[1].strip()
                results['scores'][metric] = value
    
    jobs[job_id]['compare_results'] = results

@app.route('/datasets')
def list_datasets():
    """List available proof-of-concept datasets"""
    from pathlib import Path
    
    webapp_dir = Path(__file__).parent
    zmarselo_dir = webapp_dir.parent
    
    # Available datasets (ordered from smallest to largest)
    available_datasets = [
        {
            'id': 'starwars',
            'name': 'Star Wars',
            'description': 'Star Wars character and film dataset (smallest)'
        },
        {
            'id': 'pole',
            'name': 'POLE',
            'description': 'POLE dataset'
        },
        {
            'id': 'mb6',
            'name': 'MB6',
            'description': 'MB6 dataset'
        },
        {
            'id': 'fib25',
            'name': 'FIB25',
            'description': 'FlyWire connectome dataset (25% sample)'
        },
        {
            'id': 'ldbc',
            'name': 'LDBC',
            'description': 'LDBC Social Network Benchmark (largest)'
        }
    ]
    
    # Filter datasets that have data directories
    datasets = []
    for ds in available_datasets:
        data_dir = zmarselo_dir / f"02_pgs/pg_data_{ds['id']}"
        if data_dir.exists():
            datasets.append(ds)
    
    return jsonify({'datasets': datasets})

@app.route('/ground-truth/<dataset_id>')
def get_ground_truth(dataset_id):
    """Get ground truth schema for a specific dataset"""
    # Use the correct directory path
    webapp_dir = Path(__file__).parent
    zmarselo_dir = webapp_dir.parent
    gt_dir = zmarselo_dir / '03_outputs' / 'schemas' / 'ground_truth'
    
    if not gt_dir.exists():
        return jsonify({'error': 'Ground truth schema directory not found'}), 404
    
    # Map dataset_id to ground truth file
    dataset_map = {
        'starwars': 'gt_starwars.json',
        'pole': 'gt_pole.json',
        'mb6': 'gt_mb6.json',
        'fib25': 'gt_fib25.json',
        'ldbc': 'gt_ldbc.json'
    }
    
    gt_filename = dataset_map.get(dataset_id)
    if not gt_filename:
        return jsonify({'error': f'Unknown dataset: {dataset_id}. Available datasets: {", ".join(dataset_map.keys())}'}), 404
    
    gt_file = gt_dir / dataset_id / gt_filename
    if not gt_file.exists():
        return jsonify({'error': f'Ground truth file not found: {gt_file}'}), 404
    
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_schema = json.load(f)
        return jsonify(gt_schema)
    except Exception as e:
        return jsonify({'error': f'Error loading ground truth: {str(e)}'}), 500

@app.route('/ground-truth')
def get_ground_truth_default():
    """Get ground truth schema for comparison (backwards compatibility)"""
    # Use the correct directory path
    webapp_dir = Path(__file__).parent
    zmarselo_dir = webapp_dir.parent
    gt_base_dir = zmarselo_dir / '03_outputs' / 'schemas' / 'ground_truth'
    
    if not gt_base_dir.exists():
        return jsonify({'error': 'Ground truth schema directory not found'}), 404
    
    # Find the first available ground truth JSON file
    gt_files = []
    for dataset_dir in gt_base_dir.iterdir():
        if dataset_dir.is_dir():
            for json_file in dataset_dir.glob('gt_*.json'):
                gt_files.append(json_file)
    
    if not gt_files:
        return jsonify({'error': 'No ground truth schema files found'}), 404
    
    # Load the first available ground truth file
    gt_file = gt_files[0]
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_schema = json.load(f)
        return jsonify(gt_schema)
    except Exception as e:
        return jsonify({'error': f'Error loading ground truth: {str(e)}'}), 500

def similar(a, b):
    if not a or not b: return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(name, target_list):
    best_score = 0
    best_match = None
    for target in target_list:
        score = similar(name, target)
        if score > 0.8:
            if score > best_score:
                best_score = score
                best_match = target
    return best_match

def compare_properties(gt_props, inf_props):
    gt_names = {str(p['name']) for p in gt_props if p.get('name')}
    inf_names = {str(p['name']) for p in inf_props if p.get('name')}
    return gt_names.intersection(inf_names), gt_names - inf_names, inf_names - gt_names

@app.route('/compare', methods=['POST'])
def compare_schemas():
    """Compare inferred schema with ground truth"""
    data = request.json
    gt_file = data.get('gt_file')
    inferred_file = data.get('inferred_file')
    
    if not gt_file or not inferred_file:
        return jsonify({'error': 'Both GT and inferred files required'}), 400
    
    try:
        # Load schemas
        with open(gt_file, 'r', encoding='utf-8-sig') as f:
            gt = json.load(f)
        with open(inferred_file, 'r', encoding='utf-8-sig') as f:
            inf = json.load(f)
        
        # Compare node types
        gt_nodes = {n.get('name') or (n.get('labels', [''])[0] if n.get('labels') else ''): n for n in gt.get('node_types', [])}
        inf_nodes = {n.get('name') or (n.get('labels', [''])[0] if n.get('labels') else ''): n for n in inf.get('node_types', [])}
        
        node_matches = []
        for gt_name in gt_nodes:
            match_name = find_best_match(gt_name, inf_nodes.keys())
            if match_name:
                node_matches.append({'gt': gt_name, 'inferred': match_name})
        
        # Compare properties
        total_props, total_matches = 0, 0
        property_details = []
        for match in node_matches:
            gt_name, inf_name = match['gt'], match['inferred']
            tp, fn, fp = compare_properties(
                gt_nodes[gt_name].get('properties', []),
                inf_nodes[inf_name].get('properties', [])
            )
            total_props += len(gt_nodes[gt_name].get('properties', []))
            total_matches += len(tp)
            property_details.append({
                'node': gt_name,
                'correct': list(tp),
                'missing': list(fn),
                'extra': list(fp)
            })
        
        # Compare edge types
        gt_edges = {e.get('type') or e.get('name'): e for e in gt.get('edge_types', [])}
        inf_edges = {e.get('type') or e.get('name'): e for e in inf.get('edge_types', [])}
        
        edge_matches = []
        for gt_name in gt_edges:
            match_name = find_best_match(gt_name, inf_edges.keys())
            if match_name:
                edge_matches.append({'gt': gt_name, 'inferred': match_name})
        
        accuracy = (total_matches / total_props * 100) if total_props > 0 else 0
        
        return jsonify({
            'accuracy': round(accuracy, 2),
            'node_matches': node_matches,
            'edge_matches': edge_matches,
            'property_details': property_details,
            'total_properties': total_props,
            'matched_properties': total_matches
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

