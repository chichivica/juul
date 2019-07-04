# -*- coding: utf-8 -*-
"""
Cluster embeddings and return time of their visit and departure
and list of image paths

Algorithm:
    1. Get sizes of all detected faces
    2. If a face is smaller than the minimum sizes then discard as bypasser (-1)
    3. If bigger then cluster and Calculate visit duration/times
    4. If duration is small, then discard as bypasser (-3)
    5. If frequently observed, then employer (juul or security) (-2)
    6. Else visitor (0 - num clusters)
    7. Save cluster numbers, visit and departure times
"""

from scipy.cluster import hierarchy
import sys, os
#import dlib
import numpy as np
import pandas as pd
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, load_pkl


INPUT_FILE = 'data/interim/embeddings/demo_juul_2019-07-01.pkl'
WRITE_CLUSTERS = 'data/interim/clusters/demo_juul_2019-07-01.npy'
WRITE_SEEN_TIMES = 'data/interim/clusters/demo_juul_2019-07-01_times.csv'
THRESHOLD = 0.5
MIN_CLUSTER_SIZE = 2
MIN_SPATIAL = 50
TIME_CHUNKS = 10
EMPLOYEE_CHUNKS = 7
MIN_SECONDS = 5
DISCARD_SMALL_CLUSTERS = False
DISCARD_EMPLOYEES = False
DISCARD_SHORT_TIMERS = False



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
    print(f'{len(set(cluster_labels))} unique clusters')
    return cluster_labels


def calc_sizes(boxes):
    '''
    Calculate width and height from x,y points
    '''
    boxes = np.array(boxes)
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    return np.stack([w,h], axis=1) 


def filter_small_faces(sizes, min_width, min_height):
    '''
    Get indices of boxes that satisfy the minima criteria
    '''
    sizes = np.array(sizes)
    w_ind = np.argwhere(sizes[:,0] > min_width)
    h_ind = np.argwhere(sizes[:,1] > min_height)
    intersect = np.intersect1d(w_ind, h_ind)
    print(f'Width ok {len(w_ind)}, height ok {len(h_ind)}, '
            f'intersect {len(intersect)}')
    return intersect


def small_cluster_labels(clusters, min_num):
    '''
    Find clusters with small number of members
    '''
    labs,cnts = np.unique(clusters, return_counts=True)
    small_labs = labs[np.argwhere(cnts < min_num).squeeze(1)]
    print(f'Small clusters number {len(small_labs)}')
    return small_labs


def replace_cluster_labels(clusters, labels2replace, replace_value):
    '''
    Replace cluster labels with the given value
    '''
    replace_fn = lambda x: replace_value if x in labels2replace else x
    clusters = np.array(list(map(replace_fn, clusters)))
    print(f'{len(labels2replace)} clusters were replaced by {replace_value}. '
          f'In total {sum(clusters == replace_value)}')
    return clusters


def make_splits(timestamps, num_splits):
    '''
    Make time chunks within a day for frequency count
    '''
    timestamps = np.array(timestamps)
    min_time = timestamps.min()
    max_time = timestamps.max()
    seconds = (max_time - min_time) / num_splits
    splits = np.zeros(len(timestamps))
    upper_bound = min_time + seconds
    for i in range(1, num_splits):
        splits[timestamps > upper_bound] = i
        upper_bound += seconds
    print(f'{num_splits} time chunks of len {seconds:.1f} '
          f'from {min_time} to {max_time}')
    print(f'{len(np.unique(splits))} unique splits')
    return splits


def first_last_seen(clusters, timestamps, time_splits, cond=None):
    '''
    Get time when person was seen first and last
    '''
    data = pd.DataFrame({
            'cluster': clusters,
            'time': timestamps,
            'split': time_splits,
                })
    if cond: # remove lines
        before = len(data)
        data = data.query(cond)
        after = len(data)
        print(f'{before - after} lines removed due to {cond} condition. '
              f'Now {after}')
    # get min, max time, duration  and frequency as unique splits
    seen = data.groupby('cluster', as_index=False)\
                .agg({'time': ['min', 'max', 'count'],
                      'split': 'nunique'})
    seen['duration_sec'] = seen[('time','max')] - seen[('time','min')]
    # rename colnames
    l0 = seen.columns.get_level_values(0)
    l1 = seen.columns.get_level_values(1)
    join_levels_fn = lambda x,y: '_'.join([x,y]) if y != '' else x
    new_names = list(map(join_levels_fn, l0, l1))
    seen.columns = new_names
    return seen
    

def employee_cluster_labels(clusters, frequencies, min_frequency):
    '''
    Get cluster labels related to employees as having
    high frequency
    '''
    employee_labs = clusters[np.argwhere(frequencies >= min_frequency).squeeze(1)]
    print(f'Number of employees {len(employee_labs)}')
    return employee_labs


def short_time_labels(clusters, durations, min_duration):
    '''
    Find clusters with small number of members
    '''
    short_labs = clusters[np.argwhere(durations >= min_duration).squeeze(1)]
    print(f'Number of short bypassers {len(short_labs)}')
    return short_labs


if __name__ == '__main__':
    # prepare folders
    file = get_abs_path(__file__, INPUT_FILE, depth=2)
    out = get_abs_path(__file__, WRITE_CLUSTERS, depth=2)
    out_times = get_abs_path(__file__, WRITE_SEEN_TIMES, depth=2)
    create_dir(os.path.dirname(out), False)
    # get input data
    data = load_pkl(file, True)
    sizes = calc_sizes(data['boxes'])
    large_indices = filter_small_faces(sizes, MIN_SPATIAL, MIN_SPATIAL)
    clustering_data = np.array(data['embeddings'])[large_indices]
    # cluster embeddings
    h_clusters = hierarchical_clustering(clustering_data, THRESHOLD)
#    dlib_vectors = [dlib.vector(e) for e in data['embeddings']]
#    cw_clusters = dlib.chinese_whispers_clustering(dlib_vectors, 0.4)
    # get times of first and last seen, frequencies
    h_timestamps = np.array(data['timestamps'])[large_indices]
    h_splits = make_splits(h_timestamps, TIME_CHUNKS)
    seen_times = first_last_seen(h_clusters, h_timestamps, h_splits,
                                 cond='cluster >= 0')
    if DISCARD_SMALL_CLUSTERS:
        # correct small clusters
        small_clusters = small_cluster_labels(h_clusters, MIN_CLUSTER_SIZE)
        h_clusters = replace_cluster_labels(h_clusters, small_clusters, -4)
    if DISCARD_EMPLOYEES:
        # correct employees
        employee_clusters = employee_cluster_labels(seen_times['cluster'].values,
                                                    seen_times['split_nunique'].values,
                                                    EMPLOYEE_CHUNKS)
        h_clusters = replace_cluster_labels(h_clusters, employee_clusters, -2)
        seen_times = seen_times[~seen_times['cluster'].isin(employee_clusters)]
    if DISCARD_SHORT_TIMERS:
        # correct close but fast
        short_timers = short_time_labels(seen_times['cluster'].values, 
                                         seen_times['duration_sec'], MIN_SECONDS)
        h_clusters = replace_cluster_labels(h_clusters, short_timers, -3)
    # create final labels, save clusters and time of visits
    clusters = np.zeros(len(sizes)) - 1
    clusters[large_indices] = h_clusters
    np.save(out, clusters)
    seen_times.to_csv(out_times, index=False)
    print(f'{len(set(clusters))} cluster labels saved to {out}')
    print(f'{len(seen_times)} cluster seen times saved to {out_times}')