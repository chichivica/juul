# -*- coding: utf-8 -*-
"""
Functions to load mobilenet ssd
and make inference
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from math import ceil


def get_detection_graph(path_to_graph):
    '''
    Load graph
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def graph_detections2crops(images, boxes, scores, classes, offset=0,
                           confidence_threshold=0.7):
    '''
    Convert predictions to crops
    '''
    assert len(boxes) > 0, 'Boxes are empty'
    confident = pd.DataFrame(np.argwhere(scores > confidence_threshold), 
                             columns=['row','col'])
    faces = pd.DataFrame(np.argwhere(classes == 1),
                         columns=['row','col'])
    face_indices = pd.merge(confident, faces, how='inner', on=['row','col'])
    if len(face_indices) == 0:
        return [],[],[]
    rects, confs, indices = [], [], []
    for row,col in face_indices.values:
        image = images[row]
        h,w = image.shape[:2]
        ymin, xmin, ymax, xmax = boxes[row, col]
        xmin,xmax = int(xmin * w), int(xmax * w)
        ymin,ymax = int(ymin * h), int(ymax * h)
        rects.append((xmin,ymin,xmax,ymax))
        confs.append(scores[row,col])
        indices.append((row + offset,col))
    return rects,confs,indices
    

def graph_detect_faces(detection_graph, session, images_np,
                       face_confidence=0.7):
    '''
    Run a session to detect faces on an image
    '''
    # get tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
    # some vals
#    num_images = len(images_np)
#    boxes, scores, indices = [], [], []
#    num_steps = ceil(num_images / batch_size)
#    for step in range(num_steps):
#        begin = step * batch_size
#        end = min((step + 1) * batch_size, num_images)
#        input_tensor = images_np[begin: end]
    (batch_boxes, batch_scores, batch_classes) = \
                    session.run([box_tensor, score_tensor, class_tensor],
                                  feed_dict={image_tensor: images_np})
#        boxes[begin:end,:,:] = batch_boxes
#        scores[begin:end,:] = batch_scores
#        classes[begin:end,:] = batch_classes
    batch_boxes, batch_scores, batch_indices = \
                graph_detections2crops(images_np, batch_boxes, batch_scores, 
                                       batch_classes,
                                       confidence_threshold=face_confidence)
#    boxes.extend(batch_boxes)
#    scores.extend(batch_scores)
#    indices.extend(batch_indices)
#    return boxes, scores, indices
    return batch_boxes, batch_scores, batch_indices