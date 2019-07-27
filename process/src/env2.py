# -*- coding: utf-8 -*-
"""
Set certain configs for calculation of face embeddings
and clustering
"""

import os


ENVIRON = dict(
    FACE_CONFIDENCE = os.environ.get('FACE_CONFIDENCE',0.9),
    EVERY_NTH_FRAME = os.environ.get('EVERY_NTH_FRAME', 5),
    CROP_FRAMES = {os.environ.get('FRAME_TOP', 300), 
                   os.environ.get('FRAME_BOTTOM', 1100),
                   os.environ.get('FRAME_LEFT', 0), 
                   os.environ.get('FRAME_RIGHT', 2400)},
    MIN_SPATIAL = os.environ.get('MIN_SPATIAL', 100),
    RESIZE_FRAMES = os.environ.get('RESIZE_FRAMES', None),
    BEGIN_FRAME = os.environ.get('BEGIN_FRAME', None),
    MAX_FRAMES = os.environ.get('MAX_FRAMES', None),
    BATCH_SIZE = os.environ.get('BATCH_SIZE', 16),
    DETECTOR = os.environ.get('FACE_DETECTOR', 'mtcnn'),
    RECOGNITION = os.environ.get('FACE_RECOGNITION','facenet'),
    VIDEO_PATH = os.environ.get('VIDEO_PATH', '/mnt/neurus_storage04/RG_mnt/juul_{date}/'),
    DETECTED_FACES = os.environ.get('DETECTED_FACES', 'data/interim/{name}_{detector}/'),
    TMP_DIR = os.environ.get('TMP_DIR', 'data/tmp'),
    WRITE_DETECTIONS = os.environ.get('WRITE_DETECTIONS', 'data/interim/embeddings/{name}_{detector}.hdf5'),
    WRITE_EMBEDDINGS = os.environ.get('WRITE_EMBEDDINGS','data/interim/embeddings/{name}_{recognition}.hdf5'),
    WRITE_CLUSTERS = os.environ.get('WRITE_CLUSTERS','data/interim/clusters/{name}.npy'),
    WRITE_SILHOUETTE = os.environ.get('WRITE_SILHOUETTE', 'data/interim/clusters/{name}_silhouette.csv'),
    WRITE_RESULTS = os.environ.get('WRITE_RESULTS','data/processed/clusters/{name}_{date}.csv'),
    WRITE_DEMOGRAPHICS = os.environ.get('WRITE_DEMOGRAPHICS', 'data/processed/clusters/{name}_{date}_demographics.pkl'),
    WRITE_FRAMES = os.environ.get('WRITE_FRAMES', 'data/processed/juul_photos/{name}_{date}/'),
    VIDEO_EXTENSIONS = os.environ.get('VIDEO_EXTENSIONS', 'mp4').split(','),
    RECOGNITION_MODEL_PATH = {
        os.environ.get('FACENET_PATH', 'models/20180402-114759/20180402-114759.pb'),
        os.environ.get('DLIB_RESNET_PATH', '~/dlib/dlib_face_recognition_resnet_model_v1.dat'),
            },
    SHAPE_PREDICTOR_PATH = os.environ.get('DLIB_SHAPE_PREDICTOR_PATH', 
                            '~/dlib/shape_predictor_5_face_landmarks.dat'),
    DETECTOR_WEIGHTS = {
        os.environ.get('MOBILENET_SSD_WEIGHTS', 'models/external/frozen_inference_graph_face.pb'),
        os.environ.get('MTCNN_WEIGHTS', 'src/FaceDetector/output/caffe_models/'),
        os.environ.get('DLIB_CNN_DETECTOR_WEIGHTS', '/home/neuro/dlib/mmod_human_face_detector.dat'),
                },
    DB_CONNECTION = dict(
        os.environ.get('DB_NAME', 'juuldb'),
        os.environ.get('DB_USER', 'neurus'),
        os.environ.get('DB_PASSWORD'),
        os.environ.get('DB_HOST', '172.17.0.2'),
        os.environ.get('DB_PORT', '5432'))
    )
