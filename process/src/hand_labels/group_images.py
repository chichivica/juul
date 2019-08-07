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
from src.utils import create_dir, get_abs_path, load_hdf, get_cmd_argv
from src.env import configs


if __name__ == '__main__':
    # prepare paths
    q_name = get_cmd_argv(sys.argv, 2, 'test')
    q_date = get_cmd_argv(sys.argv, 1, None)
    OUT_DIR = f'data/interim/{q_name}_clustered'
    CLUSTER_LABELS = configs['WRITE_CLUSTERS'].format(name=q_name)
    DATA_FILE = configs['WRITE_DETECTIONS'].format(name=q_name, detector=configs['DETECTOR'])
    out_dir = get_abs_path(__file__, OUT_DIR, depth=3)
    create_dir(out_dir, True)
    labels_file = get_abs_path(__file__, CLUSTER_LABELS, depth=3)
    data_file = get_abs_path(__file__, DATA_FILE, depth=3)
    # load data
    cluster_labels = np.load(labels_file)
    data = load_hdf(data_file, print_results=True)
    decode_fn = lambda x: x.decode('utf-8')
    image_paths = list(map(decode_fn, data['image_paths']))
    boxes = data['boxes']
    # save groups
    uniq_labels = list(set(cluster_labels))
    dest_dirs = [os.path.join(out_dir, str(d)) for d in uniq_labels]
    _ = [create_dir(d, True) for d in dest_dirs]
    for src,lab in tqdm(zip(image_paths, cluster_labels)):
        dest = os.path.join(out_dir, str(lab))
        subprocess.call(f'cp {src} {dest}', shell=True)
    
