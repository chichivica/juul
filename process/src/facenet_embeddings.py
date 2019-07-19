# -*- coding: utf-8 -*-
"""
Calculate face embeddings using Facenet tensorflow models

Input:
    image paths from face detection
Output:
    numpy array of shape (num_images, embedding_size)
"""

import importlib
import os,sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tqdm import tqdm
import time
import numpy as np
# custom
project_dir = os.path.dirname(os.path.dirname(__file__))
if not project_dir in sys.path: sys.path.insert(0, project_dir)

facenet = importlib.import_module('src.facenet.src.facenet', project_dir)
from src.utils import load_hdf, create_hdf, get_abs_path, get_cmd_argv
from src import env


BATCH_SIZE = 256
IMAGE_SIZE = 160


def compute_embeddings(model_path, image_paths, batch_size, image_size):
    '''
    Compute embeddings using tensorflow Facenet implementation
    '''
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    g = tf.Graph()
    inference_time = 0
    with g.as_default():
        with tf.Session(config=config) as sess:
            facenet.load_model(model_path)
            embeddings_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embeddings = np.zeros((len(image_paths), embeddings_tensor.get_shape()[1]))
            images_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            steps,remainder = divmod(len(image_paths), batch_size)
            steps += 1 if remainder > 0 else 0
            for step in tqdm(range(steps), desc='Batch inference'):
                start = step * batch_size
                end = min(len(image_paths), (step + 1) * batch_size)
                input_data = facenet.load_data(image_paths[start:end], False, False, image_size)
                time_start = time.time()
                embeddings[start:end, :] = sess.run(embeddings_tensor, 
                                                    feed_dict={images_tensor: input_data, 
                                                                    phase_train: False})
                batch_duration = time.time() - time_start
                inference_time += batch_duration
    print(f'Mean inference time {inference_time/len(image_paths):.4f}, '
                                 f'batch size {batch_size}')
    return embeddings

if __name__ == '__main__':
    # get configs
    stage = get_cmd_argv(sys.argv, 1, 'test')
    configs = env.ENVIRON[stage]
    INPUT_FILE = configs['WRITE_DETECTIONS'].format(detector=configs['DETECTOR'],
                                                    name=configs['NAME'])
    OUTPUT_FILE = configs['WRITE_EMBEDDINGS'].format(recognition=configs['RECOGNITION'],
                                                    name=configs['NAME'])
    MODEL_PATH = configs['RECOGNITION_MODEL_PATH'][configs['RECOGNITION']]
    # adjust paths
    input_file = get_abs_path(__file__, INPUT_FILE , depth=2)
    output_file = get_abs_path(__file__, OUTPUT_FILE , depth=2)
    model_path = get_abs_path(__file__, MODEL_PATH , depth=2)
    # load image paths
    image_paths = load_hdf(INPUT_FILE, keys=['image_paths'], print_results=True)['image_paths']
    image_paths = list(map(lambda x: x.decode('utf-8'), image_paths))
    # get embeddings
    embeddings = compute_embeddings(model_path, image_paths, BATCH_SIZE, IMAGE_SIZE)
    create_hdf(output_file, {'embeddings': embeddings}, print_results=True)