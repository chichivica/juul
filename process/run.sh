#!/bin/sh
# Main script

query_date=$(date -d "-$1 days" +'%Y-%m-%d')

stage=docker

python3.7 src/face_detector.py $stage $query_date
python3.7 src/facenet_embeddings.py $stage
python3.7 src/cluster_faces.py $stage
python3.7 src/write_results.py $stage

rm -rf data