# -*- coding: utf-8 -*--
"""
Detect faces on videos and create embeddings
"""

import cv2
import numpy as np
import os, sys
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from multiprocessing import Queue as mpQueue
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
import mtcnn
import dlib
import torch
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, get_file_list, \
                        mtcnn_detections2crops, dlib_detections2crops
from src.utils import create_hdf, append_to_hdf, load_hdf, get_cmd_argv
from src.mobilenet_ssd import get_detection_graph, graph_tensor_names,\
                                 graph_detections2crops
from src import env


if __name__ == '__main__':
    stage = get_cmd_argv(sys.argv, 1, default='test')
    configs = env.ENVIRON[stage]
    FACE_CONFIDENCE = configs['FACE_CONFIDENCE']
    EVERY_NTH_FRAME = configs['EVERY_NTH_FRAME']
    q_date = get_cmd_argv(sys.argv, 2, default=None)
    VIDEO_PATH = configs['VIDEO_PATH'].format(date = q_date)
    DETECTED_FACES = configs['DETECTED_FACES'].format(detector=configs['DETECTOR'],
                                                      name=configs['NAME'])
    TMP_DIR = configs['TMP_DIR']
    WRITE_DETECTIONS = configs['WRITE_DETECTIONS'].format(detector=configs['DETECTOR'],
                                                          name=configs['NAME'])
    VIDEO_EXTENSIONS = configs['VIDEO_EXTENSIONS']
    CROP_FRAMES = configs['CROP_FRAMES']
    MIN_SPATIAL = configs['MIN_SPATIAL']
    RESIZE_FRAMES = configs['RESIZE_FRAMES']
    BEGIN_FRAME = configs['BEGIN_FRAME']
    MAX_FRAMES = configs['MAX_FRAMES']
    RECOGNITION_MODEL_PATH = configs['RECOGNITION_MODEL_PATH'][configs['RECOGNITION']]
    SHAPE_PREDICTOR_PATH = configs['SHAPE_PREDICTOR_PATH']
    DETECTOR_WEIGHTS = configs['DETECTOR_WEIGHTS']
    BATCH_SIZE = configs['BATCH_SIZE']
    FILE_DEPTH = configs['FILE_DEPTH']
    DETECTOR = configs['DETECTOR']


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
            save cropped faces if required
    '''
    def __init__(self, output_dir, embeddings_path, batch_size, tmp_dir,
                 face_confidence, min_spatial, crop_frames=None, resize_frames=None,
                 skip_frames=1, stop_at_frame=None, start_at_frame=None, 
                 save_crops=True):
        self.skip_frames = skip_frames
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.face_confidence = face_confidence
        self.min_spatial = min_spatial
        self.crop_frames = crop_frames
        self.resize_frames = resize_frames
        self.save_crops = save_crops
        self.embeddings_path = embeddings_path
        self.tmp_dir = tmp_dir
        self.stop_at_frame = stop_at_frame
        self.start_at_frame = start_at_frame
        self.print_time = True
        self.debug = True
        self.queue_max = 10
        self.fps = 20
    
    
    def load_detector(self, detector, model_path):
        '''
        Load Mobilenet SSD detector as tensorflow graph
        '''
        assert os.path.exists(model_path), '{} does not exist'.format(model_path)
        assert detector in ['mtcnn', 'mobilenet_ssd','dlib'], \
                            'Detector should be mtcnn, dlib or mobilenet_ssd'
        if detector == 'mobilenet_ssd':
            self.detector = get_detection_graph(model_path)
            self.detector_tensor_names = graph_tensor_names(self.detector)
        elif detector == 'mtcnn':
            pnet, rnet, onet = mtcnn.get_net_caffe(model_path)
            pnet.share_memory()
            rnet.share_memory()
            onet.share_memory()
            self.detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')
        else:
            self.detector = dlib.cnn_face_detection_model_v1(model_path)
        self.detector_ = detector
    
    
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
#            ret, frame = cap.read()
            ret = cap.grab()
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
                ret, frame = cap.retrieve()
                if not ret:
                    if self.debug:
                        print(f'Could not read {frame_num} frame from {video_file}')
                    break
                if self.detector_ != 'mtcnn':  # pytorch-mtcnn uses bgr
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.crop_frames:
                    frame = frame[self.crop_frames['top']: self.crop_frames['bottom'],
                                  self.crop_frames['left']: self.crop_frames['right'],
                                  :]
                if self.resize_frames:
                    h,w = frame.shape[:2]
                    resize_to = (int(w * self.resize_frames), int(h * self.resize_frames))
                    frame = cv2.resize(frame, resize_to, cv2.INTER_AREA)
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
        if self.detector_ == 'mobilenet_ssd':
            (batch_boxes, batch_scores, batch_classes) = \
                self.sess.run([self.detector_tensor_names['box_tensor'], 
                          self.detector_tensor_names['score_tensor'], 
                          self.detector_tensor_names['class_tensor']
                          ], feed_dict={
                        self.detector_tensor_names['image_tensor']: frames})
            boxes, scores, indices = \
                        graph_detections2crops(frames, batch_boxes, 
                                        batch_scores, batch_classes,
                                   confidence_threshold=self.face_confidence,
                                   min_spatial=self.min_spatial)
            landmarks = None
        elif self.detector_ == 'mtcnn':
            detect_fn = lambda x: self.detector.detect(x, 
                                    threshold=[0.6,0.7,self.face_confidence], 
                                    minsize=self.min_spatial)
            detections = np.array(list(map(detect_fn, frames)))
            boxes, landmarks, indices = \
                        mtcnn_detections2crops(detections[:,0], detections[:,1])
            scores = [self.face_confidence] * len(boxes)
        else:
            detect_fn = lambda x: self.detector(x)
            detections = list(map(detect_fn, frames))
            boxes, scores, indices = dlib_detections2crops(detections)
            landmarks = None
        duration = time.time() - start_time
        if self.print_time:
            print(f'{duration:.4f} seconds for {len(frames)} frames')
        return boxes, scores, indices, landmarks


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
    def serialize_data(filepath, data, mode, chunks=True, debug=False):
        '''
        Saves embeddings with timestamps and related cropped face
        '''
        allowed = ['w','a','r']
        assert mode in allowed, f'Mode should be either of {allowed}'
        if mode == 'r':
            return load_hdf(filepath, print_results=debug)
        elif mode == 'a':
            append_to_hdf(filepath, data, print_results=debug)
        else:
            create_hdf(filepath, data, chunks=chunks, print_results=debug)
    
    
    def process_batches(self, queue, video_name, mode):
        '''
        Detect faces and compute embeddings on each batch of frames
        and save embeddings, timestamps, crops' paths and sizes
        '''
    #        self.load_recognizer(RECOGNITION_MODEL_PATH, 
    #                              shape_predictor_path=SHAPE_PREDICTOR_PATH)
        batch_num = 0
        data = {'embeddings' : [], 'image_paths': [], 'boxes': [],
                'timestamps': [], 'scores': [], 'indices': [],
                'landmarks': []}
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
            frames = self.serialize_data(path, data=None, mode='r')['batch']
            rects, scores, indices, landmarks = self.detect_faces(frames)
            if self.debug: print('Detections', len(rects))
            if len(rects) > 0:
                # get only frames of interest, convert to rgb if pytorch-mtcnn
                if self.detector_ == 'mtcnn':
                    face_images = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) \
                                   for i,_ in indices]
                else:
                    face_images = [frames[i] for i,_ in indices]
                # adjust indices to reflect video's frame number
                start_offset = 0 if self.start_at_frame is None else self.start_at_frame
                offset = batch_num * self.batch_size * self.skip_frames + start_offset
                indices = [(r * self.skip_frames + offset, c) for r,c in indices]
                assert len(rects) == len(scores) == len(indices) == len(face_images),\
                                        'Error in lengths of rects/scores/indices/images'
                # get embeddings
#                embeddings = self.compute_embeddings(face_images, rects)
                # save crops if needed
                cropped_paths = []
                if self.save_crops:
                    cropped_paths = self.save_cropped_faces(face_images, rects, indices, video_name)
                # get detection timestamps and sizes and save
                timestamps = [self.get_timestamp(int(video_name), int(f), self.fps) 
                                for f,_ in indices]
#                data['embeddings'].extend(embeddings)
                data['image_paths'].extend(cropped_paths)
                data['timestamps'].extend(timestamps)
                data['scores'].extend(scores)
                data['boxes'].extend(rects)
                data['indices'].extend(indices)
                data['landmarks'].extend(landmarks)
            batch_num += 1
            os.remove(path)
        # convert to numpy and strings to S90 for hdf5 compatibility
        data = {k:np.array(v) for k,v in data.items()}
        if len(data['image_paths']) > 0:
            data['image_paths'] = data['image_paths'].astype('S90')
        self.serialize_data(self.embeddings_path, data, mode=mode, 
                            debug=self.debug)
    
    
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
#                print('Empty queue')
                continue
            # None is a signal to break out
            if frame is None:
                if len(batch) > 0:
                    path = os.path.join(self.tmp_dir, '{}.hdf5'.format(i))
                    self.serialize_data(path, data={'batch': np.array(batch)},
                                        chunks=False,mode='w')
                    out_q.put(path)
                break
            else:
                # append to queue when batch size reached
                batch.append(frame)
                if len(batch) == self.batch_size:
                    path = os.path.join(self.tmp_dir, '{}.hdf5'.format(i))
                    self.serialize_data(path, data={'batch': np.array(batch)},
                                            chunks=False,mode='w')
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
        assert self.detector, 'Load models first'
        # define producer/consumer queues and processes
        frames_queue = Queue(maxsize = 2 * self.batch_size)
        producer = Thread(target=self.get_frames, args=(video_path, frames_queue))
        if self.detector_ == 'mtcnn':
            batches_queue = mpQueue(maxsize = self.queue_max)
            consumer = mp.Process(target=self.process_batches, 
                                  args=(batches_queue, video_name, mode))
        else:
            batches_queue = Queue(maxsize = self.queue_max)
            consumer = Thread(target=self.process_batches, 
                              args=(batches_queue, video_name, mode))
        # launch process
        producer.daemon = True
        producer.start()
        consumer.daemon = True
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
        if self.detector_ == 'mobilenet_ssd':
            with self.detector.as_default():
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.compat.v1.Session(graph=self.detector, config=config) as self.sess:
                    for i,vid in enumerate(tqdm(video_files, desc='Processing video')):
                        mode = 'w' if not file_exists else 'a'
                        try:
                            self.run_video(vid, mode)
                            file_exists = True
                        except EmptyVideoException:
                            print(f'No faces detected in {vid}')
                            continue
        else:
            for i,vid in enumerate(tqdm(video_files, desc='Processing video')):
                mode = 'w' if not file_exists else 'a'
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
    detections_filepath = get_abs_path(__file__, WRITE_DETECTIONS, depth=FILE_DEPTH)
    create_dir(os.path.dirname(detections_filepath), False)
    detector_weights = {k:get_abs_path(__file__, f, depth=FILE_DEPTH) for \
                                k,f in DETECTOR_WEIGHTS.items()}
    # create class and load face detector and recognizer
    torch.manual_seed(100)
    mp.set_start_method('spawn', force=True)
    faces = FaceEmbeddings(out_dir, detections_filepath, BATCH_SIZE, tmp_dir,
                           FACE_CONFIDENCE, min_spatial=MIN_SPATIAL,
                           skip_frames=EVERY_NTH_FRAME,
                           crop_frames=CROP_FRAMES, resize_frames=RESIZE_FRAMES,
                           start_at_frame=BEGIN_FRAME, stop_at_frame=MAX_FRAMES,
                           save_crops=True)
    faces.load_detector(DETECTOR, detector_weights[DETECTOR])
    # iterate over video-frames, detect faces and get embeddings
    video_files = get_file_list(video_dir, VIDEO_EXTENSIONS)
    faces.run(video_files)
    
#TODO
    #set fps values from a video
    #semaphor tracker