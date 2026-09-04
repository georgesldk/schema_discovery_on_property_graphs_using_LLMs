import os
import json
import threading
import time
import subprocess
from flask import Flask, render_template, request, jsonify, send_file
import sys
from pathlib import Path

# Add src to path so we can import pg_schema_llm
webapp_dir = Path(__file__).parent
src_dir = webapp_dir.parent / "src"
sys.path.insert(0, str(src_dir))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global state for job tracking
jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-dataset/<dataset_id>', methods=['POST'])
def process_dataset(dataset_id):
    """Process a Neo4j-backed dataset using infer.py."""
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
                    elif 'Asking Mistral' in line:
                        jobs[job_id]['progress'] = 60
                        jobs[job_id]['message'] = 'Calling Mistral API...'
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

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'message': job.get('message', ''),
        'progress': job.get('progress', 0),
        'dataset_id': job.get('dataset_id'),
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

