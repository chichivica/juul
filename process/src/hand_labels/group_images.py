# -*- coding: utf-8 -*-
"""
Get all images and their clusters and copy them
into grouped folders in order for hand labelling
"""

import sys, os
import numpy as np
from tqdm import tqdm
import subprocess
#custom
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, load_pkl


OUT_DIR = 'data/interim/clustered_images'
CLUSTER_LABELS = 'data/interim/clusters/demo_juul_2019-07-01.npy'
DATA_FILE = 'data/interim/embeddings/demo_juul_2019-07-01.pkl'

if __name__ == '__main__':
    # prepare paths
    out_dir = get_abs_path(__file__, OUT_DIR, depth=3)
    create_dir(out_dir, True)
    labels_file = get_abs_path(__file__, CLUSTER_LABELS, depth=3)
    data_file = get_abs_path(__file__, DATA_FILE, depth=3)
    # load data
    cluster_labels = np.load(labels_file)
    data = load_pkl(DATA_FILE, True)
    image_paths = data['image_paths']
    boxes = data['boxes']
    # save groups
    uniq_labels = list(set(cluster_labels))
    dest_dirs = [os.path.join(out_dir, str(d)) for d in uniq_labels]
    _ = [create_dir(d, True) for d in dest_dirs]
    for src,lab in tqdm(zip(image_paths, cluster_labels)):
        dest = os.path.join(out_dir, str(lab))
        subprocess.call(f'cp {src} {dest}', shell=True)
    