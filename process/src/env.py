# -*- coding: utf-8 -*-
"""
Set certain configs for calculation of face embeddings
and clustering
"""

import os


configs = dict(
    REQUIRE_PRINTS = eval(os.environ.get('REQUIRE_PRINTS', 'True')),
    FACE_CONFIDENCE = float(os.environ.get('FACE_CONFIDENCE',0.9)),
    EVERY_NTH_FRAME = int(os.environ.get('EVERY_NTH_FRAME', 5)),
    CROP_FRAMES = {
        'top': int(os.environ.get('FRAME_TOP', 300)), 
        'bottom': int(os.environ.get('FRAME_BOTTOM', 1100)),
        'left': int(os.environ.get('FRAME_LEFT', 0)), 
        'right': int(os.environ.get('FRAME_RIGHT', 2400)),
        },
    MIN_SPATIAL = int(os.environ.get('MIN_SPATIAL', 100)),
    RESIZE_FRAMES = eval(os.environ.get('RESIZE_FRAMES', 'None')),
    BEGIN_FRAME = eval(os.environ.get('BEGIN_FRAME', 'None')),
    MAX_FRAMES = eval(os.environ.get('MAX_FRAMES', 'None')),
    BATCH_SIZE = int(os.environ.get('DETECTOR_BATCH_SIZE', 16)),
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
    WRITE_DEMOGRAPHICS = os.environ.get('WRITE_DEMOGRAPHICS', 'data/processed/clusters/{name}_{date}_demographics.csv'),
    WRITE_FRAMES = os.environ.get('WRITE_FRAMES', 'data/processed/juul_photos/{name}_{date}/'),
    VIDEO_EXTENSIONS = os.environ.get('VIDEO_EXTENSIONS', 'mp4').split(','),
    RECOGNITION_MODEL_PATH = {
        'facenet': os.environ.get('FACENET_PATH', 
                                  'models/20180402-114759/20180402-114759.pb'),
        'dlib': os.environ.get('DLIB_RESNET_PATH', 
                               '~/dlib/dlib_face_recognition_resnet_model_v1.dat'),
            },
    SHAPE_PREDICTOR_PATH = os.environ.get('DLIB_SHAPE_PREDICTOR_PATH', 
                            '~/dlib/shape_predictor_5_face_landmarks.dat'),
    DETECTOR_WEIGHTS = {
        'mobilenet_ssd': os.environ.get('MOBILENET_SSD_WEIGHTS', 
                                        'models/external/frozen_inference_graph_face.pb'),
        'mtcnn': os.environ.get('MTCNN_WEIGHTS', 
                                'src/FaceDetector/output/caffe_models/'),
        'dlib': os.environ.get('DLIB_CNN_DETECTOR_WEIGHTS', 
                               '/home/neuro/dlib/mmod_human_face_detector.dat'),
                },
    DEMOGRAPHIC = {
        'vggface': os.environ.get('VGGFACE_WEIGHTS',
                                  'models/external/resnet50_ft_weight.pkl'),
        'vggface_backbone': os.environ.get('VGGFACE_BACKBONE', 'resnet'),
        'age': os.environ.get('AGE_CLASSIFIER_WEIGHTS',
                              'models/age_gender/best/vgg_age_cv{index}.hdf'),
        'age_out_classes': int(os.environ.get('AGE_OUT_CLASSES', '8')),
        'gender': os.environ.get('GENDER_CLASSIFIER_WEIGHTS',
                                 'models/age_gender/best/vgg_gender_cv{index}.hdf'),
        'gender_out_classes': int(os.environ.get('GENDER_OUT_CLASSES', '1')),
        'num_folds': int(os.environ.get('DEMOGRAPHIC_CV_FOLDS', '5')),
        'input_size': eval(os.environ.get('DEMOGRAPHIC_MODEL_INPUT_SIZE',
                                          '(224,224)')),
        'batch_size': int(os.environ.get('DEMOGRAPHIC_BATCH_SIZE', '64')),
            },
    DB_CONNECTION = dict(
        dbname = os.environ.get('DB_NAME', 'juuldb'),
        user = os.environ.get('DB_USER', 'neurus'),
        password = os.environ.get('DB_PASSWORD', ''),
        host = os.environ.get('DB_HOST', '172.17.0.3'),
        port = os.environ.get('DB_PORT', '5432'),)
    )
    
assert configs['DB_CONNECTION']['password'] != '', 'Set DB_PASSWORD as environment variable'
