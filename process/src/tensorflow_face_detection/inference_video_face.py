#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys, os
import time
import numpy as np
import tensorflow as tf
import cv2

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


from src.tensorflow_face_detection.utils import label_map_util
from src.tensorflow_face_detection.utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/external/frozen_inference_graph_face.pb'
PATH_TO_CKPT = os.path.join(project_dir, PATH_TO_CKPT )

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'src/tensorflow_face_detection/protos/face_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(project_dir, PATH_TO_LABELS )

VIDEO_PATH = "data/raw/2019-06-21/1561110713.mp4"
VIDEO_PATH = os.path.join(project_dir, VIDEO_PATH )

VIDEO_OUT = "src/tensorflow_face_detection/test_out.avi"
VIDEO_OUT = os.path.join(project_dir, VIDEO_OUT )


NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490;
    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter(VIDEO_OUT, fourcc, 20.0, (w, h))


      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print(f'inference time: {elapsed_time:.4f}, num detections: {num_detections[0]:.4f}')
      print(boxes.shape,)
      print(scores.shape,)
      print(classes.shape,)
#      print(num_detections)
      exit()
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      out.write(image)


    cap.release()
    out.release()
