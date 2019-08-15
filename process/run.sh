#!/bin/sh
# Main script

query_date=$(date -d "-$1 days" +'%Y-%m-%d')

stage=$2

/usr/local/bin/python3.7 /root/people_count/src/face_detector.py $query_date $stage
/usr/local/bin/python3.7 /root/people_count/src/cluster_faces.py $query_date $stage
/usr/local/bin/python3.7 /root/people_count/src/predict_age_gender.py $query_date $stage
/usr/local/bin/python3.7 /root/people_count/src/retrieve_frames.py $query_date $stage
/usr/local/bin/python3.7 /root/people_count/src/write_results.py $query_date $stage

rm -rf data
