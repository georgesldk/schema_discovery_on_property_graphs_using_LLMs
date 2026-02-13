# Dataset Used so far
fib25 , pole , starwars , mb6 , ldbc

# How to extract its GT
python scripts/extract_gt.py datasetName

# How to infer its schema
python scripts/infer.py datasetName

# How to compare infer with GT
python scripts/compare.py datasetName

# How to run webapp
python app.py 

# Example Performance of Outputs
STARWARS							  100%
POLE						    95%
MB6						  	    66%
FIB25						    68%


# How to get the datasets
 - visit https://zenodo.org/records/17801336
 - Get ground truth data (.pgs)
 - Get whole Property Graph files (.csv)
 - In 01_gts create folder named gt_data_<datasetName>
    - Put the ground truth data here
 - In 02_pts create folder named pt_data_<datasetName>
    - Put the Property Graph files here

- Navigate to /Zmarselo/ and run commands
- Or navigate to /Zmarselo/webapp and run webapp