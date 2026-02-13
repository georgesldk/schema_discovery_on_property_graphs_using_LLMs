# All commands should be run from the root /Zmarselo/ directory.

# How to extract its GT
python scripts/extract_gt.py datasetName

# How to infer its schema
python scripts/infer.py datasetName

# How to compare infer with GT
python scripts/compare.py datasetName

# Launch Web Interface from /Zmarselo/webapp/ directory
python app.py 



# Dataset Used so far
fib25 , pole , starwars , mb6 , ldbc

# Example Performance of Outputs
STARWARS					    96.18%
POLE						    95.83%
MB6						  	    65.89%
FIB25						    69.90%
LDBC                            92.91%


# How to get the datasets
 - visit https://zenodo.org/records/17801336
 - Get ground truth data (.pgs)
 - Get whole Property Graph files (.csv)
 - In 01_gts create folder named gt_data_<datasetName>
    - Put the ground truth data here
 - In 02_pts create folder named pt_data_<datasetName>
    - Put the Property Graph files here


# If you encounter ModuleNotFoundError or shell errors:

# Ensure conda is deactivated and your local venv is active:
source venv/bin/activate

# If scripts cannot find the src folder, export the Python path: 
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Complete informative report in PDF