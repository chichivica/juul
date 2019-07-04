# -*- coding: utf-8 -*-
"""
Get new clusters after re-grouping face images by hand
and save them to use in making video
"""

import sys, os
import numpy as np
import pandas as pd
import importlib
#custom
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, load_pkl, write_dict
from src.rgutils.nlp_utils import get_class_files
#from cluster_demographics import cluster_demographics
cd = importlib.import_module('cluster_demographics', os.path.realpath(__file__))
importlib.reload(cd)


IMAGES_DIR = 'data/interim/clustered_images_corrected'
WRITE_LABELS = 'data/interim/clusters/corrected_demo_juul_2019-07-01.npy'
DATA_FILE = 'data/interim/embeddings/demo_juul_2019-07-01.pkl'
WRITE_DATA = 'data/interim/embeddings/corrected_demo_juul_2019-07-01.pkl'
WRITE_DEMOGRAPHICS = 'data/interim/clusters/demographics_demo_juul_2019-07-01.pkl'
REMOVE_FP = True

labels_mapping = {
        'small': -1,
        'false_positives': -5,
        'employee': -2,
        }

def merge_image_paths(orig_paths, cluster_dict):
    corrected = pd.DataFrame()
    for lab,imgs in cluster_dict.items():
        tmp = pd.DataFrame({
                    'cluster': [lab] * len(imgs),
                    'image_path': imgs
                    })
        corrected = corrected.append(tmp)
    corrected['image_name'] = corrected['image_path']\
                                .apply(lambda x: os.path.basename(x))
    orig_names = pd.DataFrame({
                'image_name': [os.path.basename(p) for p in orig_paths]
                })
    print(f'Orig len {len(orig_names)}, corrected len {len(corrected)}')
    orig_names['image_name'] = orig_names['image_name'].astype(str)
    corrected['image_name'] = corrected['image_name'].astype(str)
    corrected = orig_names.merge(corrected, how='left', on='image_name')
    clusters = corrected['cluster'].replace(labels_mapping).astype(int).values
    return clusters


def select_data(data, indices):
    '''
    Selects indices from data and returns shortened
    '''
    new_data = {}
    for k,v in data.items():
        new_v = list(np.array(v)[indices])
        new_data[k] = new_v
    return new_data

if __name__ == '__main__':
    # prepare paths
    images_dir = get_abs_path(__file__, IMAGES_DIR, depth=3)
    write_labels = get_abs_path(__file__, WRITE_LABELS, depth=3)
    create_dir(os.path.dirname(write_labels), False)
    data_file = get_abs_path(__file__, DATA_FILE, depth=3)
    write_data = get_abs_path(__file__, WRITE_DATA, depth=3)
    create_dir(os.path.dirname(write_data), False)
    write_demographics = get_abs_path(__file__, WRITE_DEMOGRAPHICS, depth=3)
    create_dir(os.path.dirname(write_demographics), False)
    # load image paths
    data = load_pkl(data_file)
    image_paths = data['image_paths']
    # get filenames by folder name
    cluster_labels = os.listdir(images_dir)
    cluster_images = get_class_files(images_dir, cluster_labels)
    # merge to order cluster labels
    correct_labels = merge_image_paths(image_paths, cluster_images)
    # remove false positives
    if REMOVE_FP:
        tp_indices = np.argwhere(correct_labels != labels_mapping['false_positives'])\
                        .squeeze(1)
        correct_labels = correct_labels[tp_indices]
        data = select_data(data, tp_indices)
    # save
    np.save(write_labels, correct_labels)
    print(f'{len(set(correct_labels))} corrected cluster labels saved to {write_labels}')
    write_dict(write_data, data, False)
    write_dict(write_demographics, cd.cluster_demographics, False)
