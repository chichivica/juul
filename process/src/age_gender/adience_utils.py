# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import skimage
import cv2
from tqdm import tqdm


def get_image_path(table_row, data_dir):
    '''
    Get image path and correspoing landmarks for a row in labels table
    '''
    img_fname = '.'.join(['coarse_tilt_aligned_face', str(table_row['face_id']), table_row['original_image']])
    lmk_fname = img_fname.replace('coarse_tilt_aligned_face', 'landmarks')
    lmk_fname = lmk_fname.replace('jpg', 'txt')
    img_path = os.path.join(data_dir, table_row['user_id'], img_fname)
    lmk_path = os.path.join(data_dir, table_row['user_id'], lmk_fname)
    return img_path, lmk_path


def load_labels(labels_dir, indices):
    '''
    Load label files by index
    '''
    fname = 'fold_{index}_data.txt'
    labels = pd.DataFrame()
    for i in indices:
        path = os.path.join(labels_dir, fname.format(index = i))
        tmp = pd.read_table(path, usecols=['user_id','original_image','face_id','tilt_ang','age','gender'])
        tmp['fold'] = i
        labels = labels.append(tmp)
    return labels


def facial_keypoints(image_paths, keypoint_paths, num_random=None, 
                     keypoints=True, crop=False, save_dir=None):
    images = []
    if num_random:
        indices = np.random.choice(range(len(image_paths)),  size=num_random)
        image_paths = np.array(image_paths)[indices]
        keypoint_paths = np.array(keypoint_paths)[indices]
    for ip,kp in tqdm(zip(image_paths, keypoint_paths)):
        img = skimage.io.imread(ip)
        lmk = pd.read_table(kp, skiprows=1, sep=',')
        if keypoints:
            for _,row in lmk.iterrows():
                loc = (int(row['x_c']),int(row['y_c']))
                cv2.circle(img, loc, radius=3, color=(255,0,0), thickness=3)
        if crop:
            left,top,right,bottom = keypoint_bounds(lmk['x_c'], lmk['y_c'])
            img = img[top:bottom, left:right, :]
        if save_dir:
            save_path = os.path.join(save_dir, os.path.basename(ip))
            skimage.io.imsave(save_path, img)
            images.append(save_path)
        else:
            images.append(img)
    return images


def keypoint_bounds(x_values, y_values):
    '''
    From an array of x and y
    Return: left,top,right,bottom
    '''
    left = int(np.min(x_values))
    top = int(np.min(y_values))
    right = int(np.max(x_values))
    bottom = int(np.max(y_values))
    return left,top,right,bottom
