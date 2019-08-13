#!/bin/sh
# Main script

query_date=$(date -d "-$1 days" +'%Y-%m-%d')

stage=$2

python3.7 src/face_detector.py $query_date $stage
python3.7 src/cluster_faces.py $query_date $stage
python3.7 src/predict_age_gender.py $query_date $stage
python3.7 src/retrieve_frames.py $query_date $stage
python3.7 src/write_results.py $query_date $stage

rm -rf data