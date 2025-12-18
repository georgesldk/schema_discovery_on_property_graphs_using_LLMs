import os
import json
import random
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from build_graph import build_graph

# Import comparison functions
from difflib import SequenceMatcher

# Import functions from main.py - reuse existing code
import importlib.util
main_path = os.path.join(os.path.dirname(__file__), 'scripts', 'main.py')
spec = importlib.util.spec_from_file_location("main_module", main_path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Use the existing profiling functions from main.py (same code, no duplication)
profile_node_type = main_module.profile_node_type
profile_edge_type = main_module.profile_edge_type
extract_json = main_module.extract_json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size (increased for large datasets)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('schema_found', exist_ok=True)
os.makedirs('gt_schema', exist_ok=True)

# Global state for job tracking
jobs = {}

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
        jobs[job_id]['progress'] = 60
        jobs[job_id]['status'] = 'generating_prompt'
        jobs[job_id]['message'] = 'Generating schema inference prompt...'
        
        # Ask API - same prompt as main.py
        prompt = f"""
    You are a Data Architect. Reverse-engineer the schema from this profile.
    
    DATA PROFILE:
    {context_report}
    
    INSTRUCTIONS:
    1. Identify Node Types and Edge Types.
    2. Infer properties and valid data types (String, Long, Double, Boolean).
    3. Determine 'mandatory' (true if property found in >90% of nodes/edges).
    4. Generalize names (e.g. 'Neuron_Connections' -> 'CONNECTS_TO').
    5. 'roiInfo' is a JSON String.
    
    OUTPUT JSON:
    {{ "node_types": [...], "edge_types": [...] }}
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

@app.route('/upload', methods=['POST'])
def upload_files():
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
        'upload_dir': upload_dir
    }
    
    # Start processing in background thread
    output_dir = os.path.join('schema_found', job_id)
    os.makedirs(output_dir, exist_ok=True)
    thread = threading.Thread(target=process_schema_discovery, args=(upload_dir, output_dir, job_id))
    thread.daemon = True
    thread.start()
    
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
        'progress': job.get('progress', 0)
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
    app.run(debug=True, host='0.0.0.0', port=5000)

