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