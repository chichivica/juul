#!/bin/sh
# Main script

query_date=$(date -d "-$2 days" +'%Y-%m-%d')

stage=$1

python3.7 src/face_detector.py $stage $query_date
python3.7 src/cluster_faces.py $stage $query_date
python3.7 src/retrieve_frames.py $stage $query_date
python3.7 src/write_results.py $stage $query_date

rm -rf data