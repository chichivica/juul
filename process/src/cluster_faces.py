# -*- coding: utf-8 -*-
"""
Cluster embeddings and create analytics
"""
from scipy.cluster import hierarchy
import pickle
import sys, os
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path


FILE = 'data/interim/embeddings/juul_mobilenet_dlib_2019-06-26.pkl'
THRESHOLD = 0.55


def hierarchical_clustering(features, threshold, dist_type='euclidean',
                            linkage_method='complete',):
    '''
    Cluster customer features until no points that are close enough
    to form a new cluster
    '''
    link_matrix = hierarchy.linkage(features, metric=dist_type, 
                                    method=linkage_method)
    cluster_labels = hierarchy.fcluster(link_matrix, threshold, 
                                        criterion='distance')
    return cluster_labels

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    file = get_abs_path(__file__, FILE, depth=2)
    data = load_pkl(file)
    clusters = hierarchical_clustering(data['embeddings'], THRESHOLD)
    