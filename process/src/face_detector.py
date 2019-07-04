# -*- coding: utf-8 -*--
"""
Detect faces on videos and create embeddings
"""

import cv2
import numpy as np
import os, sys
from tqdm import tqdm
import pickle
import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
#from mtcnn.mtcnn import MTCNN
import dlib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, get_file_list
from src.mobilenet_ssd import get_detection_graph, graph_tensor_names,\
                                 graph_detections2crops


FACE_CONFIDENCE = 0.69
EVERY_NTH_FRAME = 1
#VIDEO_PATH = 'data/raw/2019-06-21/'
VIDEO_PATH = 'data/external/juul_stream/1556725808.mp4'
#VIDEO_PATH = 'data/external/juul_stream/'
DETECTED_FACES = 'data/interim/demo/'
TMP_DIR = 'data/tmp'
WRITE_EMBEDDINGS = 'data/interim/embeddings/demo_juul_2019-07-01.pkl'
VIDEO_EXTENSIONS = ['mp4']
CROP_FRAMES = {'top': 470, 'bottom': 1520-250,
               'left': 400, 'right': 2688-300}
BEGIN_FRAME = None
MAX_FRAMES = None
RECOGNITION_MODEL_PATH = '/home/neuro/dlib/dlib_face_recognition_resnet_model_v1.dat'
SHAPE_PREDICTOR_PATH = '/home/neuro/dlib/shape_predictor_5_face_landmarks.dat'
GRAPH_CONFIG = 'models/external/frozen_inference_graph_face.pb'
BATCH_SIZE = 32
FILE_DEPTH = 2


class EmptyVideoException(Exception):
    pass

class NoFaceDetectedException(Exception):
    pass


class FaceEmbeddings:
    '''
    Class to detect faces in videos and get their representations.
    Mobilenet SSD used for detection and Dlib used for embeddings.
    -------
    Usage:
        init this class
        load models
        for each video:
            load frames
            detect faces
            get face embeddings
            save cropped faces and detection video if required
    '''
    def __init__(self, output_dir, embeddings_path, batch_size, tmp_dir,
                 face_confidence, crop_frames=None, skip_frames=1, stop_at_frame=None,
                 start_at_frame=None, save_crops=True):
        self.skip_frames = skip_frames
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.face_confidence = face_confidence
        self.crop_frames = crop_frames
        self.save_crops = save_crops
        self.embeddings_path = embeddings_path
        self.tmp_dir = tmp_dir
        self.stop_at_frame = stop_at_frame
        self.start_at_frame = start_at_frame
        self.print_time = True
        self.debug = True
    
    def load_detector(self, model_path):
        '''
        Load Mobilenet SSD detector as tensorflow graph
        '''
        assert os.path.exists(model_path), '{} does not exist'.format(model_path)
        self.detector = get_detection_graph(graph_config)
        self.detector_tensor_names = graph_tensor_names(self.detector)
        
    def load_recognizer(self, model_path, shape_predictor_path=None):
        '''
        Load Dlib's face recognition and shape predictor models
        '''
        assert os.path.exists(model_path), '{} does not exist'.format(model_path)
        self.recognizer = dlib.face_recognition_model_v1(model_path)
        if shape_predictor_path:
            assert os.path.exists(model_path), \
                            '{} does not exist'.format(shape_predictor_path)
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def get_frames(self, video_file, queue):
        '''
        Read frames from a video file and return a list of frames
        with a skip factor
        '''
        assert os.path.exists(video_file), '{} does not exist'.format(video_file)
        cap = cv2.VideoCapture(video_file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # if unable to read then quit and add last batch to queue
            if not ret:
                if self.debug:
                    print(f'Could not read {frame_num} frame from {video_file}')
                break
            # check if frame number within boundaries if provided
            if self.start_at_frame is not None:
                if frame_num < self.start_at_frame: 
                    frame_num += 1
                    continue
            # take every n-th frame
            if frame_num % self.skip_frames == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.crop_frames:
                    frame = frame[self.crop_frames['top']: self.crop_frames['bottom'],
                                  self.crop_frames['left']: self.crop_frames['right'],
                                  :]
                queue.put(frame)
            frame_num += 1
            # if over break out
            if self.stop_at_frame is not None:
                if frame_num > self.stop_at_frame: break
        queue.put(None)
        if frame_num == 0:
            raise EmptyVideoException

        
    def detect_faces(self, frames,):
        '''
        Detect faces by batches of frames
        Select faces with high confidence
        '''
        frames = np.array(frames)
        # process frames
        start_time = time.time()
        (batch_boxes, batch_scores, batch_classes) = \
            self.sess.run([self.detector_tensor_names['box_tensor'], 
                      self.detector_tensor_names['score_tensor'], 
                      self.detector_tensor_names['class_tensor']
                      ], feed_dict={
                    self.detector_tensor_names['image_tensor']: frames})
        boxes, scores, indices = \
                    graph_detections2crops(frames, batch_boxes, 
                                    batch_scores, batch_classes,
                               confidence_threshold=self.face_confidence)
        duration = time.time() - start_time
        if self.print_time:
            print(f'{duration:.4f} seconds for {len(frames)} frames')
        return boxes, scores, indices

    def compute_embeddings(self, images, rectangle_points,):
        '''
        Prepare image for dlib's face recognition model:
            1. create rectangle object
            2. determine shape with shape predictor
            3. get face chips
            4. get embeddings
        '''
        convert2dlib_fn = lambda x: dlib.rectangle(*x)
        calc_shape_fn = lambda x,y: self.shape_predictor(x, y)
        face_chip_fn = lambda x,y: dlib.get_face_chip(x, y)
        start_time = time.time()
        dlib_rects = list(map(convert2dlib_fn, rectangle_points))
        shapes = list(map(calc_shape_fn, images, dlib_rects))
        face_chips = list(map(face_chip_fn, images, shapes))
        embeddings = self.recognizer.compute_face_descriptor(face_chips,
                                                            num_jitters=10)
        duration = time.time() - start_time
        if self.print_time:
            print(f'{duration:.4f} sec for {len(embeddings)} embeddings')
        return np.array(embeddings)
    
    def save_cropped_faces(self, images, rects, indices, video_name):
        '''
        Save cropped faces if needed
        '''
        assert os.path.exists(self.output_dir), \
                            '{} does not exist'.format(self.output_dir)
        filepaths = []
        for i,(frame_num,detect_num) in enumerate(indices):
            filename = '_'.join([video_name, str(frame_num), str(detect_num)])
            filepath = os.path.join(self.output_dir, filename + '.jpg')
            left,top,right,bottom = rects[i]
            crop = cv2.cvtColor(images[i][top:bottom, left:right], 
                                cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, crop)
            filepaths.append(filepath)
        return filepaths

        
    @staticmethod        
    def get_timestamp(start, frame_num, fps):
        '''
        Calculate timestamp of a crop based on starting timestamp,
        its frame number and fps
        '''
        for var in [start, frame_num, fps]:
            assert isinstance(var, (int,float)), f'{var} is neither int nor float'
        return int(start + (frame_num / fps))
        
        
    @staticmethod
    def serialize_data(filepath, data, mode, debug=False):
        '''
        Saves embeddings with timestamps and related cropped face
        '''
        assert mode in ['wb','ab', 'rb'], 'Mode to pickler either ab,rb or wb'
        if mode == 'rb':
            with open(filepath, 'rb') as f:
                pkl_file = pickle.load(f)
            return pkl_file
        elif mode == 'ab':
            with open(filepath, 'rb') as f:
                pkl_file = pickle.load(f)
            for k,v in data.items():
                pkl_file[k].extend(v)
            with open(filepath, 'wb') as f:
                pickle.dump(pkl_file, f)
                if debug: print('Dumped', len(pkl_file[k]), mode)
        else:
            with open(filepath, mode) as f:
                pickle.dump(data, f)
                if debug: print('Dumped', len(data[list(data.keys())[0]]), mode)
    
    def process_batches(self, queue, video_name, mode):
        '''
        Detect faces and compute embeddings on each batch of frames
        and save embeddings, timestamps, crops' paths and sizes
        '''
        batch_num = 0
        data = {'embeddings' : [], 'image_paths': [], 'boxes': [],
                'timestamps': [], 'scores': [], 'indices': []}
        while True:
            # if no items in queue just wait, None is a signal to break out
            try:
                path = queue.get(False)
            except QueueEmpty:
                continue
            if path is None:
                if self.debug: print('Breaking out of consumer')
                break
            # detect faces and get respective frames
            if self.debug:
                print('Remaining in queue', queue.qsize(), 'batch number', batch_num)
            frames = self.serialize_data(path, data=None, mode='rb')
            rects, scores, indices = self.detect_faces(frames)
            if self.debug: print('Detections', len(rects))
            if len(rects) > 0:
                face_images = [frames[i] for i,_ in indices]
                # adjust indices to reflect video's frame number
                start_offset = 0 if self.start_at_frame is None else self.start_at_frame
                offset = batch_num * self.batch_size * self.skip_frames + start_offset
                indices = [(r * self.skip_frames + offset, c) for r,c in indices]
                assert len(rects) == len(scores) == len(indices) == len(face_images),\
                                        'Error in lengths of rects/scores/indices/images'
                # get embeddings
                embeddings = self.compute_embeddings(face_images, rects)
                # save crops if needed
                cropped_paths = []
                if self.save_crops:
                    cropped_paths = self.save_cropped_faces(face_images, rects, indices, video_name)
                # get detection timestamps and sizes and save
                timestamps = [self.get_timestamp(int(video_name), int(f), self.fps) 
                                for f,_ in indices]
                data['embeddings'].extend(list(embeddings))
                data['image_paths'].extend(cropped_paths)
                data['timestamps'].extend(timestamps)
                data['scores'].extend(scores)
                data['boxes'].extend(rects)
                data['indices'].extend(indices)
            batch_num += 1
            os.remove(path)
        self.serialize_data(self.embeddings_path, data, 
                            mode=mode, debug=self.debug)
    
    def frames2batch(self, in_q, out_q,):
        '''
        From a queue of frames recieved from producer process
        creates a queue of batches for a consumer process
        '''
        batch = []
        i = 0
        while True:
            # wait for items
            try:
                frame = in_q.get(False)
            except QueueEmpty:
                continue
            # None is a signal to break out
            if frame is None:
                if len(batch) > 0:
                    path = os.path.join(self.tmp_dir, '{}.pkl'.format(i))
                    self.serialize_data(path, data=batch, mode='wb')
                    out_q.put(path)
                break
            else:
                # append to queue when batch size reached
                batch.append(frame)
                if len(batch) == self.batch_size:
                    path = os.path.join(self.tmp_dir, '{}.pkl'.format(i))
                    self.serialize_data(path, data=batch, mode='wb')
                    i += 1
                    out_q.put(path)
                    if self.debug: print('in', in_q.qsize(), 'out', out_q.qsize())
                    batch = []
        out_q.put(None)
    
    def run_video(self, video_path, mode):
        '''
        Run 1 video pipeline
        '''
        base_name = os.path.basename(video_path)
        video_name = base_name.split('.')[0]
        assert self.detector and self.recognizer, 'Load models first'
        # define producer queue and process
        frames_queue = Queue()
        producer = Thread(target=self.get_frames, args=(video_path, frames_queue))
        producer.setDaemon(True)
        producer.start()
        # define consumner queue and process
        batches_queue = Queue()
        consumer = Thread(target=self.process_batches, 
                          args=(batches_queue, video_name, mode))
        consumer.setDaemon(True)
        consumer.start()
        # run main process
        self.frames2batch(frames_queue, batches_queue)
        # wait to finish
        producer.join()
        consumer.join()
    
    def run(self, video_files):
        '''
        Run pipeline for all videos in the list
        '''
        file_exists = False
        with self.detector.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.compat.v1.Session(graph=self.detector, config=config) as self.sess:
                for i,vid in enumerate(tqdm(video_files, desc='Processing video')):
                    mode = 'wb' if not file_exists else 'ab'
                    try:
                        self.run_video(vid, mode)
                        file_exists = True
                    except EmptyVideoException:
                        print(f'No faces detected in {vid}')
                        continue
        print(f"Embeddings and auxillary data saved to {self.embeddings_path}")
    
    
if __name__ == '__main__':
    # prepare directories
    video_dir = get_abs_path(__file__, VIDEO_PATH, depth=FILE_DEPTH)
    out_dir = get_abs_path(__file__, DETECTED_FACES, depth=FILE_DEPTH)
    tmp_dir = get_abs_path(__file__, TMP_DIR, depth=FILE_DEPTH)
    create_dir(out_dir, True)
    create_dir(tmp_dir, True)
    embedding_filepath = get_abs_path(__file__, WRITE_EMBEDDINGS, depth=FILE_DEPTH)
    create_dir(os.path.dirname(embedding_filepath), False)
    graph_config = get_abs_path(__file__, GRAPH_CONFIG, depth=FILE_DEPTH)
    # create class and load face detector and recognizer
    faces = FaceEmbeddings(out_dir, embedding_filepath, BATCH_SIZE, tmp_dir,
                           FACE_CONFIDENCE, skip_frames=EVERY_NTH_FRAME,
                           crop_frames=CROP_FRAMES, start_at_frame=BEGIN_FRAME,
                           save_crops=True, stop_at_frame=MAX_FRAMES)
    faces.load_detector(graph_config)
    faces.load_recognizer(RECOGNITION_MODEL_PATH, 
                          shape_predictor_path=SHAPE_PREDICTOR_PATH)
    # iterate over video-frames, detect faces and get embeddings
    video_files = get_file_list(video_dir, VIDEO_EXTENSIONS)
    faces.run(video_files)
    
