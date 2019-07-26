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

import sys, os
import dlib
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import create_dir, get_abs_path, load_hdf, get_cmd_argv, create_hdf
from src import env
from src.facenet_embeddings import get_facenet_paths, compute_embeddings


CW_THRESHOLD = 0.95
SILHOUETTE_THRESHOLD = 0.15
STEP_SIZE = 5000
EUCLIDEAN_THRESHOLD = 0.85
MIN_CLUSTER_SIZE = 3
MIN_WIDTH = 55
MIN_HEIGHT = 65
TIME_CHUNKS = 22
EMPLOYEE_CHUNKS = TIME_CHUNKS // 3
MIN_SECONDS = 15
DISCARD_SMALL_CLUSTERS = True
DISCARD_EMPLOYEES = True
DISCARD_SHORT_TIMERS = True
DISCARD_LOW_SILHOUETTE = True
MIN_BLUR_VAR = 80
MAX_BLUR_VAR = 450
BATCH_SIZE = 256
IMAGE_SIZE = 160


def dlib_chinese_whispers(array_of_features, threshold):
    '''
    Chinese whispers clustering using Dlib library
    '''
    emb_dlib = [dlib.vector(e) for e in array_of_features]
    cw_labels = dlib.chinese_whispers_clustering(emb_dlib, threshold)
    return np.array(cw_labels)


def inter_cluster_distance(x, y):
    '''
    Get a mean distance and std for two clusters
    subsetting from all distances
    '''
    pdist = pairwise_distances(x, y)
    return pdist.mean(), pdist.std()



def multi_stage_clustering(all_features, step_size, min_cluster_size, 
                           pairwise_threshold, cw_threshold):
    '''
    Performs multi-stage clustering when number of points is too high
    to fit in memory
    '''
    steps,remainder = divmod(len(all_features), step_size)
    steps = steps + 1 if remainder > 0 else steps
    previous_subset = None
    previous_labels = None
    paired = {}
    clusters = []
    for s in tqdm(range(steps), desc='clustering subsets'):
        start,end = s * step_size, min(step_size * (s + 1), len(all_features))
        # cluster subset
        subset = all_features[start: end]
        labels = dlib_chinese_whispers(subset, cw_threshold)
        if s > 0:
            # compare mean distances to previous subsets for large clusters
            labels = labels + previous_labels.max() + 1
            prev_labs,prev_cnts = np.unique(previous_labels, return_counts=True)
            labs,cnts = np.unique(labels, return_counts=True)
            prev_labs = prev_labs[prev_cnts >= min_cluster_size]
            labs = labs[cnts >= min_cluster_size]
            if len(prev_labs) > 0 and len(labs) > 0:
                inter_cluster = [(pl,l, *inter_cluster_distance(previous_subset[previous_labels==pl],
                                                                subset[labels==l]))
                                 for pl,l in product(prev_labs, labs)]
                matched_pairs = list(filter(lambda x: x[2] < pairwise_threshold, inter_cluster))
                if len(matched_pairs) > 0:
                    paired[s] = matched_pairs
        # copy current subset for next iteration
        previous_subset = subset.copy()
        previous_labels = labels.copy()
        clusters.extend(labels)
    return paired, clusters


def reassign_labels(clusters, paired):
    '''
    Iterate over matches to get new clusters
    If multiple matches within 1 period then
    let them merge into one cluster
    '''
    keys_reverse = sorted(paired.keys(), reverse=True)
    clusters = np.array(clusters)
    for key in keys_reverse:
        matches = paired[key]
        for new_lab,old_lab,_,_ in matches:
            clusters[clusters == old_lab] = new_lab
    return clusters


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
    small_labs = labs[cnts < min_num]
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
    short_labs = clusters[np.argwhere(durations < min_duration).squeeze(1)]
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
    
    
def select_high_silhouette(embeddings, labels, silh_threshold, save_scores=None):
    '''
    Get cluster labels with high silhouette score
    and select their indices from labels array
    '''
    print('Calculating silhouette scores, might take a few minutes..')
    silh_scores = silhouette_samples(embeddings, labels)
    silh_df = pd.DataFrame({
                'score': silh_scores,
                'cluster': labels,
                })
    cluster_scores = silh_df.groupby('cluster').agg({'score': ['mean','count']})
    if save_scores:
        cluster_scores.to_csv(save_scores, index=True)
        print(f'{len(cluster_scores)} silh.scores saved to {save_scores}')
    high_silh_clusters = \
        cluster_scores.loc[cluster_scores[('score', 'mean')] > silh_threshold]\
                        .index.values
    isin_fn = lambda x: x in high_silh_clusters
    is_target = np.array(list(map(isin_fn, labels)))
    print(f'{len(high_silh_clusters)} clusters have silhouette score above {silh_threshold} '
          f'corresponding to {is_target.sum()} out of {len(is_target)} values')
    return is_target


def test_counts(final_labels, correct_len):
    '''
    Test correctness of clusters
    '''
    labs,cnts = np.unique(final_labels, return_counts=True)
    for lab,cnt in zip(labs,cnts):
        if lab < 0:
            cor = correct_len[lab]
            assert cnt == cor, f'Mismatch @ {lab}, it is {cnt} but has to be {cor}'


def first_last_image(clusters, seen_times, timestamps, image_paths, boxes):
    '''
    For each image find a path to first and last image
    also get bounding boxes
    '''
    assert len(clusters) == len(timestamps) == len(image_paths)
    first_images,last_images = [],[]
    start_boxes,end_boxes = [],[]
    for i,row in seen_times.iterrows():
        clust_ind = np.argwhere(clusters == row['cluster']).squeeze(1)
        min_time_ind = timestamps[clust_ind].argmin()
        max_time_ind = timestamps[clust_ind].argmax()
        first_image = image_paths[clust_ind][min_time_ind]
        last_image = image_paths[clust_ind][max_time_ind]
        first_images.append(first_image),last_images.append(last_image)
        begin_box = tuple(boxes[clust_ind][min_time_ind])
        end_box = tuple(boxes[clust_ind][max_time_ind])
        start_boxes.append(begin_box),end_boxes.append(end_box)
    seen_times['first_image'] = first_images
    seen_times['last_image'] = last_images
    seen_times['first_box'] = start_boxes
    seen_times['last_box'] = end_boxes
    assert seen_times.isna().any().sum() == 0
    return seen_times
    

if __name__ == '__main__':
    # get stage
    stage = get_cmd_argv(sys.argv, 1, 'test')
    q_date = get_cmd_argv(sys.argv, 2, None)
    configs = env.ENVIRON[stage]
    INPUT_DATA = configs['WRITE_EMBEDDINGS'].format(recognition=configs['RECOGNITION'],
                                                    name=configs['NAME'])
    INPUT_FILE = configs['WRITE_DETECTIONS'].format(detector=configs['DETECTOR'],
                                                    name=configs['NAME'])
    WRITE_CLUSTERS = configs['WRITE_CLUSTERS'].format(name=configs['NAME'])
    WRITE_RESULTS = configs['WRITE_RESULTS'].format(name=configs['NAME'],
                                                    date=q_date)
    WRITE_SILHOUETTE = configs['WRITE_SILHOUETTE'].format(name=configs['NAME'])
    # prepare folders
    input_data = get_abs_path(__file__, INPUT_DATA, depth=2)
    file = get_abs_path(__file__, INPUT_FILE, depth=2)
    out = get_abs_path(__file__, WRITE_CLUSTERS, depth=2)
    out_silhouette = get_abs_path(__file__, WRITE_SILHOUETTE, depth=2)
    out_results = get_abs_path(__file__, WRITE_RESULTS, depth=2)
    create_dir(os.path.dirname(out), False)
    create_dir(os.path.dirname(out_results), False)
    # get input data
    data = load_hdf(file, print_results=True)
    decode_fn = lambda x: x.decode('utf-8')
    image_paths = np.array(list(map(decode_fn, data['image_paths'])))
    timestamps = data['timestamps']
    test_lens = {}
    # remove small crops
    sizes = calc_sizes(data['boxes'])
    target_indices = filter_small_faces(sizes, MIN_WIDTH, MIN_HEIGHT)
    test_lens[-1] = len(sizes) - len(target_indices)
    # remove non-frontal images
    landmarks = data['landmarks'][target_indices]
    is_frontal = select_frontal_faces(landmarks)
    nonfrontal_indices = np.setdiff1d(target_indices,target_indices[is_frontal])
    target_indices = target_indices[is_frontal]
    test_lens[-6] = len(landmarks) - len(target_indices)
    # remove too low and too high blur variance images
    target_images = image_paths[target_indices]
    is_normal = select_blur_normal(target_images, MIN_BLUR_VAR, MAX_BLUR_VAR)
    abnormal_blur_indices = np.setdiff1d(target_indices,target_indices[is_normal])
    target_indices = target_indices[is_normal]
    test_lens[-5] = len(target_images) - len(target_indices)
    # load or calculate embeddings
    if os.path.exists(input_data):
        embeddings = load_hdf(input_data, print_results=True)['embeddings']
        embeddings = embeddings[target_indices]
    else:
        print(f'{input_data} not found. Calculating embeddings...')
        target_images = image_paths[target_indices]
        _,write_emb,model_path = get_facenet_paths(configs)
        embeddings = compute_embeddings(model_path, target_images, 
                                        BATCH_SIZE, IMAGE_SIZE)
        create_hdf(write_emb, {'embeddings': embeddings}, print_results=True)
    # cluster embeddings
    if len(embeddings) > STEP_SIZE:
        paired,clusters = multi_stage_clustering(embeddings, STEP_SIZE, 
                                                 MIN_CLUSTER_SIZE, EUCLIDEAN_THRESHOLD,
                                                 CW_THRESHOLD)
        clusters = reassign_labels(clusters, paired)
    else:
        clusters = dlib_chinese_whispers(embeddings, CW_THRESHOLD)
    # select clusters with high silhouette score
    if DISCARD_LOW_SILHOUETTE:
        is_target = select_high_silhouette(embeddings, clusters, 
                                           SILHOUETTE_THRESHOLD, out_silhouette)
        low_silh_indices = target_indices[~is_target]
        target_indices = target_indices[is_target]
        clusters = clusters[is_target]
        test_lens[-7] = len(low_silh_indices)
    # get times of first and last seen, frequencies
    target_timestamps = timestamps[target_indices]
    splits = make_splits(target_timestamps, TIME_CHUNKS)
    seen_times = first_last_seen(clusters, target_timestamps, splits, cond='cluster >= 0')
    if DISCARD_SMALL_CLUSTERS:
        # correct small clusters
        small_clusters = small_cluster_labels(clusters, MIN_CLUSTER_SIZE)
        clusters = replace_cluster_labels(clusters, small_clusters, -4)
        seen_times = seen_times[~seen_times['cluster'].isin(small_clusters)]
        test_lens[-4] = (clusters == -4).sum()
    if DISCARD_EMPLOYEES:
        # correct employees
        employee_clusters = employee_cluster_labels(seen_times['cluster'].values,
                                                    seen_times['split_nunique'].values,
                                                    EMPLOYEE_CHUNKS)
        clusters = replace_cluster_labels(clusters, employee_clusters, -2)
        seen_times = seen_times[~seen_times['cluster'].isin(employee_clusters)]
        test_lens[-2] = (clusters == -2).sum()
    if DISCARD_SHORT_TIMERS:
        # correct close but fast
        short_timers = short_time_labels(seen_times['cluster'].values, 
                                         seen_times['duration_sec'], MIN_SECONDS)
        clusters = replace_cluster_labels(clusters, short_timers, -3)
        seen_times = seen_times[~seen_times['cluster'].isin(short_timers)]
        test_lens[-3] = (clusters == -3).sum()
    # create final labels, save clusters and time of visits
    final_clusters = np.zeros(len(sizes)) - 1 # for small images
    final_clusters[target_indices] = clusters # clusters
    final_clusters[abnormal_blur_indices] = -5
    final_clusters[nonfrontal_indices] = -6
    if DISCARD_LOW_SILHOUETTE: final_clusters[low_silh_indices] = -7
    test_counts(final_clusters, test_lens)
    uniq_clusters = len(set(final_clusters[final_clusters >= 0]))
    assert len(seen_times) == uniq_clusters, f'Number of clusters in np and csv do not match'
    np.save(out, final_clusters)
    seen_times = first_last_image(final_clusters, seen_times, timestamps, 
                                  image_paths, data['boxes'])
    seen_times.to_csv(out_results, index=False)
    print(f'{uniq_clusters} cluster labels saved to {out}')
    print(f'{uniq_clusters} cluster seen times saved to {out_results}')