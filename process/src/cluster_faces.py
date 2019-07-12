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
import dlib
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, load_pkl
from src import env

if __name__ == '__main__':
    try:
        stage = sys.argv[1]
    except IndexError:
        stage = 'test'
    assert stage in env.ENVIRON.keys(), f'{stage} is not in {env.ENVIRON.keys()}'
    configs = env.ENVIRON[stage]    
    INPUT_FILE = configs['WRITE_EMBEDDINGS'].format(detector=configs['DETECTOR'])
    WRITE_CLUSTERS = configs['WRITE_CLUSTERS'].format(detector=configs['DETECTOR'])
    WRITE_SEEN_TIMES = configs['WRITE_SEEN_TIMES'].format(detector=configs['DETECTOR'])

CLUSTERING_THRESHOLD = 0.45
SILHOUETTE_THRESHOLD = 0.15
MIN_CLUSTER_SIZE = 3
MIN_WIDTH = 55
MIN_HEIGHT = 65
TIME_CHUNKS = 10
EMPLOYEE_CHUNKS = TIME_CHUNKS // 2 + 1
MIN_SECONDS = 10
DISCARD_SMALL_CLUSTERS = True
DISCARD_EMPLOYEES = True
DISCARD_SHORT_TIMERS = True
DISCARD_LOW_SILHOUETTE = True
MIN_BLUR_VAR = 80
MAX_BLUR_VAR = 400


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
    print(f'{num_splits} time chunks of len {seconds:.1f}s '
          f'from {min_time} to {max_time}')
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


def face_frontal(face_landmarks):
    '''
    Determine if face is frontal by looking at a nose position
    Considering landmarks as x,y in order: left eye, right eye,
    nose, left mouth edge, right mouth edge
    '''
    l_eye,r_eye,nose,l_mouth,r_mouth = face_landmarks 
    left_edge = max(l_eye[0], l_mouth[0])
    top_edge = max(l_eye[1], r_eye[1])
    right_edge = min(r_eye[0], r_mouth[0])
    bottom_edge = min(l_mouth[1], r_mouth[1])
    if (left_edge < nose[0] < right_edge) and \
        (top_edge < nose[1] < bottom_edge):
        return True
    else:
        return False

def select_frontal_faces(landmarks_list):
    '''
    Mark if face is frontal or not for a list of face landmarks
    '''
    frontal_fn = lambda x: face_frontal(x)
    is_frontal = np.array(list(map(frontal_fn, landmarks)))
    print(f'{is_frontal.sum()} out of {len(is_frontal)} are frontal')
    return np.argwhere(is_frontal).squeeze(1)


def calc_bluriness(image_path):
    '''
    Calculate blurriness variance for a gray image
    '''
    path = os.path.join(project_dir, image_path)
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_blur_normal(image_paths, min_blur, max_blur):
    '''
    Select indices of images that have normal blur-variance value
    '''
    blur_fn = lambda f: calc_bluriness(f)
    blur_index = np.array(list(map(blur_fn, image_paths)))
    is_normal = np.argwhere((blur_index > min_blur) & 
                            (blur_index < max_blur)).squeeze(1)
    print(f'{len(is_normal)} out of {len(image_paths)} have normal blur variance '
          f'between {min_blur} and {max_blur}')
    return is_normal


def dlib_chinese_whispers(array_of_features, threshold):
    '''
    Convert to dlib vectors and perform chinese whispering clustering
    '''
    emb_dlib = [dlib.vector(e) for e in array_of_features]
    cw_labels = dlib.chinese_whispers_clustering(emb_dlib, threshold)
    return np.array(cw_labels)


def cluster_results(features, labels):
    '''
    Results of clustering
    '''
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noize = (labels == -1).sum()
    score = silhouette_score(features, labels)
    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noizy points: {n_noize/len(labels):.3f}')
    print(f'Silhouette score: {score}')
    
    
def select_high_silhouette(embeddings, labels, silh_threshold):
    '''
    Get cluster labels with high silhouette score
    and select their indices from labels array
    '''
    silh_scores = silhouette_samples(embeddings, labels)
    silh_df = pd.DataFrame({
                'score': silh_scores,
                'cluster': labels,
                })
    cluster_scores = silh_df.groupby('cluster').agg({'score': ['mean','count']})
    high_silh_clusters = \
        cluster_scores.loc[cluster_scores[('score', 'mean')] > silh_threshold]\
                        .index.values
    isin_fn = lambda x: x in high_silh_clusters
    is_target = np.array(list(map(isin_fn, labels)))
    print(f'{len(high_silh_clusters)} clusters have silhouette score above {silh_threshold} '
          f'corresponding to {is_target.sum()} out of {len(is_target)} values')
    return is_target


if __name__ == '__main__':
    # prepare folders
    file = get_abs_path(__file__, INPUT_FILE, depth=2)
    out = get_abs_path(__file__, WRITE_CLUSTERS, depth=2)
    out_times = get_abs_path(__file__, WRITE_SEEN_TIMES, depth=2)
    create_dir(os.path.dirname(out), False)
    # get input data
    data = load_pkl(file, True)
    sizes = calc_sizes(data['boxes'])
    target_indices = filter_small_faces(sizes, MIN_WIDTH, MIN_HEIGHT)
    # remove non-frontal images
    landmarks = np.array(data['landmarks'])[target_indices]
    frontal_indices = select_frontal_faces(landmarks)
    target_indices = target_indices[frontal_indices]
    nonfrontal_indices = target_indices[np.setdiff1d(np.arange(len(target_indices)),frontal_indices)]
    # remove too low and too high blur variance images
    image_paths = np.array(data['image_paths'])[target_indices]
    normal_indices = select_blur_normal(image_paths, MIN_BLUR_VAR, MAX_BLUR_VAR)
    target_indices = target_indices[normal_indices]
    abnormal_blur_indices = target_indices[np.setdiff1d(np.arange(len(target_indices)),frontal_indices)]
    # cluster embeddings
    embeddings = np.array(data['embeddings'])[target_indices]
    clusters = dlib_chinese_whispers(embeddings, CLUSTERING_THRESHOLD)
    cluster_results(embeddings, clusters)
    # select clusters with high silhouette score
    if DISCARD_LOW_SILHOUETTE:
        is_target = select_high_silhouette(embeddings, clusters, SILHOUETTE_THRESHOLD)
        target_indices = target_indices[is_target]
        clusters = clusters[is_target]
    # get times of first and last seen, frequencies
    timestamps = np.array(data['timestamps'])[target_indices]
    splits = make_splits(timestamps, TIME_CHUNKS)
    seen_times = first_last_seen(clusters, timestamps, splits,
                                 cond='cluster >= 0')
    if DISCARD_SMALL_CLUSTERS:
        # correct small clusters
        small_clusters = small_cluster_labels(clusters, MIN_CLUSTER_SIZE)
        clusters = replace_cluster_labels(clusters, small_clusters, -4)
    if DISCARD_EMPLOYEES:
        # correct employees
        employee_clusters = employee_cluster_labels(seen_times['cluster'].values,
                                                    seen_times['split_nunique'].values,
                                                    EMPLOYEE_CHUNKS)
        clusters = replace_cluster_labels(clusters, employee_clusters, -2)
    if DISCARD_SHORT_TIMERS:
        # correct close but fast
        short_timers = short_time_labels(seen_times['cluster'].values, 
                                         seen_times['duration_sec'], MIN_SECONDS)
        clusters = replace_cluster_labels(clusters, short_timers, -3)
    # create final labels, save clusters and time of visits
    final_clusters = np.zeros(len(sizes)) - 1
    final_clusters[target_indices] = clusters
    final_clusters[abnormal_blur_indices] = -5
    final_clusters[nonfrontal_indices] = -6
    np.save(out, final_clusters)
    seen_times.to_csv(out_times, index=False)
    print(f'{len(set(clusters))} cluster labels saved to {out}')
    print(f'{len(seen_times)} cluster seen times saved to {out_times}')