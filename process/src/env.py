# -*- coding: utf-8 -*-
"""
Set certain configs for calculation of face embeddings
and clustering
"""

ENVIRON = {
   'test': dict(
            FACE_CONFIDENCE = 0.9,
            EVERY_NTH_FRAME = 5,
            CROP_FRAMES = {'top': 470, 'bottom': 1520-250,
                           'left': 400, 'right': 2688-300},
            MIN_SPATIAL = 55,
            RESIZE_FRAMES = None,
            BEGIN_FRAME = None,
            MAX_FRAMES = None,
            BATCH_SIZE = 32,
            DETECTOR = 'mtcnn',
            FILE_DEPTH = 2,
            VIDEO_PATH = 'data/external/juul_short/1556725808.mp4',
#            VIDEO_PATH = 'data/external/juul_stream/',
            DETECTED_FACES = 'data/interim/autodemo_{detector}/',
            TMP_DIR = 'data/tmp',
            WRITE_EMBEDDINGS = 'data/interim/embeddings/autodemo_{detector}.pkl',
            WRITE_CLUSTERS = 'data/interim/clusters/autodemo_{detector}.npy',
            WRITE_SEEN_TIMES = 'data/interim/clusters/autodemo_times_{detector}.csv',
            WRITE_DEMOGRAPHICS = 'data/interim/clusters/autodemo_demographics_{detector}.pkl',
            VIDEO_EXTENSIONS = ['mp4'],
            RECOGNITION_MODEL_PATH = '/home/neuro/dlib/dlib_face_recognition_resnet_model_v1.dat',
            SHAPE_PREDICTOR_PATH = '/home/neuro/dlib/shape_predictor_5_face_landmarks.dat',
            DETECTOR_WEIGHTS = {
                        'mobilenet_ssd': 'models/external/frozen_inference_graph_face.pb',
                        'mtcnn': 'src/FaceDetector/output/caffe_models/',
                        'dlib': '/home/neuro/dlib/mmod_human_face_detector.dat',
                        },
            ),
       }