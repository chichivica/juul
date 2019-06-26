# -*- coding: utf-8 -*-
"""
Utility functions 
"""

import os, shutil
import cv2


def create_dir(dir_path, empty_if_exists=True):
    '''
    Creates a folder if not exists,
    If does exist, then empties it by default
    '''
    if os.path.exists(dir_path):
        if empty_if_exists:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)
        
        

def get_abs_path(src, dest, depth=2):
    '''
    Get correct absolute path by going deeper from source to find project dir
    And join it with dest
    '''
    project_dir = os.path.abspath(src)
    for i in range(depth):
        project_dir = os.path.dirname(project_dir)
    return os.path.join(project_dir, dest)


def get_file_list(path, extensions):
    '''
    Get list of files from a folder that match given extensions
    If a file given, then return it is a single-element list
    '''
    assert os.path.exists(path), '{} does not exist'.format(path)
    if os.path.isdir(path):
        files = os.listdir(path)
        files = [os.path.join(path, f) for f in files \
                if f.split('.')[-1] in extensions]
        assert len(files) > 0, f'No files in {path} that are in allowed format: {extensions}'
        return files
    else:
        assert path.split('.')[-1] in extensions, f'{path} is not allowed format: {extensions}'
        return [path]
    
    
def draw_rectangles(image, rects, confidences):
    '''
    Draw one or multiple rectangles
    '''
    for c,rect in enumerate(rects):
        p1 = tuple(rect[:2])
        p2 = tuple(rect[2:])
        cv2.rectangle(image, p1, p2, (0,255,0), 2)
        conf = confidences[c]
        cv2.putText(image, f'Face {conf:.2f}', p1, cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (0,0,255), 1)
    
def detections2crops(detections, frame):
    '''
    Get a list of detections from MTCNN and return list of crops
    '''
    crops = []
    rects = []
    for detected in detections:
        left,top,w,h = detected['box']
        right = left + w
        bottom = top + h
        cropped = frame[top:bottom, left:right]
        crops.append(cropped), rects.append((left,top,right,bottom))
    return crops, rects
