# -*- coding: utf-8 -*-
"""
Functions to load mobilenet ssd
and make inference
"""
import tensorflow as tf
import numpy as np
import pandas as pd


def get_detection_graph(path_to_graph):
    '''
    Load graph with weights
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def graph_detections2crops(images, boxes, scores, classes, offset=0,
                           confidence_threshold=0.7, min_spatial=50):
    '''
    Convert graph predictions to lists of boxes, confidences and indices
    '''
    assert len(boxes) > 0, 'Boxes are empty'
    # get indices of high scores and face classes, inner join
    confident = pd.DataFrame(np.argwhere(scores > confidence_threshold), 
                             columns=['row','col'])
    faces = pd.DataFrame(np.argwhere(classes == 1),
                         columns=['row','col'])
    face_indices = pd.merge(confident, faces, how='inner', on=['row','col'])
    if len(face_indices) == 0:
        return [],[],[] # return empty if None
    # create 3 lists to return 
    rects, confs, indices = [], [], []
    for row,col in face_indices.values:
        image = images[row]
        h,w = image.shape[:2]
        ymin, xmin, ymax, xmax = boxes[row, col]
        xmin,xmax = int(xmin * w), int(xmax * w)
        ymin,ymax = int(ymin * h), int(ymax * h)
        if ymax - ymin < min_spatial or xmax - xmin < min_spatial:
            continue
        rects.append((xmin,ymin,xmax,ymax))  # rectangels
        confs.append(scores[row,col]) # face probabilities
        indices.append((row + offset,col)) # tuples (frame number, detection  number)
    return rects,confs,indices


def graph_tensor_names(detection_graph):
    '''
    Return tensor names for trainer MobileNet SSD
    '''
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
    return {'image_tensor':image_tensor, 'box_tensor':box_tensor,
            'score_tensor':score_tensor, 'class_tensor':class_tensor}
    
